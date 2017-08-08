
import tensorflow as tf
from layers import lrelu, deconv

ROTATION_COUNT = 8
EPS = 1e-12


class Model:

    def __init__(self, opt, depth, width):
        self.opt = opt
        self.depth = depth
        self.width = width
        self.create_model()

    def create_model(self):
        self.inputs = tf.placeholder(tf.float32, shape=(self.opt.batch_size, self.width, self.width, self.depth))
        self.targets = tf.placeholder(tf.float32, shape=(self.opt.batch_size, self.width, self.width, self.depth))

        true_tensor = tf.constant([1], dtype=tf.float32)
        false_tensor = tf.constant([0], dtype=tf.float32)

        inside = [0] * self.depth
        inside[-1] = 1
        inside[self.depth - ROTATION_COUNT - 1] = 1
        inside_tensor = tf.constant(inside, dtype=tf.float32)

        flat_inputs = tf.reshape(self.inputs, shape=(self.opt.batch_size * self.width * self.width, self.depth))
        inside_mask = tf.map_fn(lambda row: tf.cond(
            tf.equal(
                tf.reduce_prod(tf.cast(tf.equal(row, inside_tensor), tf.int32)), 1),
            lambda: true_tensor,
            lambda: false_tensor), flat_inputs)
        inside_mask = tf.reshape(inside_mask, shape=(self.opt.batch_size, self.width, self.width, 1))
        outside_mask = 1 - inside_mask

        with tf.variable_scope("generator"):
            self.outputs, category, rotation = self.create_generator(self.inputs)

        with tf.name_scope("real_discriminator"):
            with tf.variable_scope("discriminator"):
                self.predict_real = self.create_discriminator(self.inputs, self.targets)

        with tf.name_scope("fake_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                self.predict_fake = self.create_discriminator(self.inputs, self.outputs)

        eps = tf.random_uniform([self.opt.batch_size, 1], minval=0., maxval=1.)
        X_inter = eps * self.targets + (1. - eps) * self.outputs

        with tf.name_scope("interpolate_discriminator"):
            with tf.variable_scope("discriminator", reuse=True):
                predict_interpolate = self.create_discriminator(self.inputs, X_inter)

        grad = tf.gradients(predict_interpolate, [X_inter])[0]
        grad_norm = tf.sqrt(tf.reduce_sum(grad ** 2, axis=1))
        self.grad_pen = self.opt.lam * tf.reduce_mean((grad_norm - 1.) ** 2)

        with tf.name_scope("discriminator_adversarial_loss"):
            # predict_real => 1
            # predict_fake => 0
            # self.discrim_GAN_loss = tf.reduce_mean(predict_fake) - tf.reduce_mean(predict_real)
            # self.discrim_loss = self.discrim_GAN_loss + self.grad_pen
            self.discrim_loss = tf.reduce_mean(-(tf.log(self.predict_real + EPS) + tf.log(1 - self.predict_fake + EPS)))

        with tf.name_scope("generator_adversarial_loss"):
            # predict_fake => 1
            # self.gen_loss_GAN = -tf.reduce_mean(predict_fake) * self.opt.gan_weight
            self.gen_loss_GAN = tf.reduce_mean(-tf.log(self.predict_fake + EPS)) * self.opt.gan_weight

            target_category = self.targets[:, :, :, :self.depth - ROTATION_COUNT]
            target_rotation = self.targets[:, :, :, self.depth - ROTATION_COUNT:]
            self.category_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=category * outside_mask + target_category * inside_mask,
                                                        labels=target_category))
            self.rotation_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=rotation * outside_mask + target_rotation * inside_mask,
                                                        labels=target_rotation))
            self.gen_loss_L1 = (self.category_loss + self.rotation_loss) * self.opt.l1_weight
            self.gen_loss = self.gen_loss_GAN + self.gen_loss_L1

            self.gen_supervised_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=rotation,
                                                        labels=target_rotation)
            ) + tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=category,
                                                        labels=target_category)
            )

        self.global_step = tf.contrib.framework.get_or_create_global_step()
        self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)

        with tf.name_scope("discriminator_adversarial_train"):
            discrim_tvars = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
            discrim_optim = tf.train.AdamOptimizer(self.opt.lr_discriminator, self.opt.beta1)
            self.discrim_grads_and_vars = discrim_optim.compute_gradients(self.discrim_loss, var_list=discrim_tvars)
            self.discrim_train = discrim_optim.apply_gradients(self.discrim_grads_and_vars)

        with tf.name_scope("generator_adversarial_train"):
            with tf.control_dependencies([self.incr_global_step]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(self.opt.lr_generator, self.opt.beta1)
                self.gen_grads_and_vars = gen_optim.compute_gradients(self.gen_loss, var_list=gen_tvars)
                self.gen_adversarial_train = gen_optim.apply_gradients(self.gen_grads_and_vars)

        with tf.name_scope("generator_supervised_train"):
            with tf.control_dependencies([self.incr_global_step]):
                gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
                gen_optim = tf.train.AdamOptimizer(self.opt.lr_generator, self.opt.beta1)
                gen_grads_and_vars = gen_optim.compute_gradients(self.gen_supervised_loss, var_list=gen_tvars)
                self.gen_supervised_train = gen_optim.apply_gradients(gen_grads_and_vars)

    def create_discriminator(self, room, design):
        n_layers = 4
        layers = []

        input = tf.concat([room, design], axis=3)
        layers.append(input)

        for i in range(n_layers):
            with tf.variable_scope("layer_%d" % len(layers)):
                out_channels = self.opt.ndf * 2**i
                convolved = tf.layers.conv2d(layers[-1], filters=out_channels, kernel_size=2, strides=2, padding='valid')
                normalized = tf.layers.batch_normalization(convolved)
                rectified = lrelu(normalized, 0.2)
                layers.append(rectified)

        with tf.variable_scope("layer_%d" % len(layers)):
            output = tf.layers.conv2d(layers[-1], filters=1, kernel_size=2, strides=1, padding='valid')
            output = tf.nn.sigmoid(output)
            layers.append(output)

        return layers[-1]

    def create_generator(self, room):
        layers = []

        # encoder_1: [batch, 32, 32, in_channels] => [batch, 16, 16, ngf]
        with tf.variable_scope("encoder_1"):
            output = tf.layers.conv2d(room, filters=self.opt.ngf, kernel_size=2, strides=2, padding='valid')
            layers.append(output)

        layer_specs = [
            self.opt.ngf * 2,  # encoder_2: [batch, 16, 16, ngf] => [batch, 8, 8, ngf * 2]
            self.opt.ngf * 4,  # encoder_3: [batch, 8, 8, ngf * 2] => [batch, 4, 4, ngf * 4]
            self.opt.ngf * 8,  # encoder_4: [batch, 4, 4, ngf * 4] => [batch, 2, 2, ngf * 8]
            self.opt.ngf * 16,
        ]

        for out_channels in layer_specs:
            with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
                rectified = lrelu(layers[-1], 0.2)
                # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
                convolved = tf.layers.conv2d(rectified, filters=out_channels, kernel_size=2, strides=2, padding='valid')
                output = tf.layers.batch_normalization(convolved)
                layers.append(output)

        layer_specs = [
            (self.opt.ngf * 8, 0.1),
            (self.opt.ngf * 4, 0.1),  # decoder_8: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 4]
            (self.opt.ngf * 2, 0.1),  # decoder_7: [batch, 4, 4, ngf * 4 * 2] => [batch, 8, 8, ngf * 2]
            (self.opt.ngf * 1, 0.1),  # decoder_6: [batch, 8, 8, ngf * 2 * 2] => [batch, 16, 16, ngf * 1]
        ]

        num_encoder_layers = len(layers)
        for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
                if decoder_layer == 0:
                    # first decoder layer doesn't have skip connections
                    # since it is directly connected to the skip_layer
                    input = layers[-1]
                else:
                    input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

                rectified = tf.nn.relu(input)
                # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
                output = deconv(rectified, out_channels)
                output = tf.layers.batch_normalization(output)

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

                layers.append(output)

        # decoder_1: [batch, 16, 16, ngf * 2] => [batch, 32, 32, generator_outputs_channels]
        with tf.variable_scope("decoder_1"):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = deconv(rectified, self.depth)

            category = output[:, :, :, :self.depth - ROTATION_COUNT]
            category_output = tf.nn.softmax(category)

            rotation = output[:, :, :, self.depth - ROTATION_COUNT:]
            rotation_output = tf.nn.softmax(rotation)

            final_output = tf.concat([category_output, rotation_output], axis=3)
            layers.append(final_output)

        return layers[-1], category, rotation

    def pre_train_G(self, f, sess, loader, times):
        fetches = f.copy()
        fetches['train'] = self.gen_supervised_train
        for _ in range(times):
            rooms, layouts = loader.next_batch(0)
            results = sess.run(fetches, {self.inputs: rooms, self.targets: layouts})
        return results

    def pre_train_D(self, f, sess, loader, times):
        fetches = f.copy()
        fetches['train'] = self.discrim_train
        for _ in range(times):
            rooms, layouts = loader.next_batch(0)
            sess.run(fetches, {self.inputs: rooms, self.targets: layouts})



