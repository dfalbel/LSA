encoder <- torch::nn_module(
  initialize = function(input_shape, code_length) {

    self$input_shape = input_shape
    self$code_length = code_length

    c(c, h, w) %<-% input_shape

    activation_fn <- torch::nn_leaky_relu()

    # Convolutional network
    self$conv = torch::nn_sequential(
      downsample_block(channel_in=c, channel_out=32, act=activation_fn),
      downsample_block(channel_in=32, channel_out=64, act=activation_fn),
    )

    self$deepest_shape <- c(64, h %/% 4, w %/% 4)

    # FC network
    self$fc = torch::nn_sequential(
      torch::nn_linear(in_features=prod(self$deepest_shape), out_features=64),
      torch::nn_batch_norm1d(num_features=64),
      activation_fn,
      torch::nn_linear(in_features=64, out_features=code_length),
      torch::nn_sigmoid()
    )
  },
  forward = function(x) {
    x %>%
      self$conv() %>%
      torch::torch_flatten(start_dim = 2) %>%
      self$fc()
  }
)

decoder <- torch::nn_module(
  initialize = function(code_length, deepest_shape, output_shape) {

    self$code_length <- code_length
    self$deepest_shape <- deepest_shape
    self$output_shape <- output_shape

    activation_fn = torch::nn_leaky_relu()

    # FC network
    self$fc <- torch::nn_sequential(
      torch::nn_linear(in_features=code_length, out_features=64),
      torch::nn_batch_norm1d(num_features=64),
      activation_fn,
      torch::nn_linear(in_features=64, out_features=prod(deepest_shape)),
      torch::nn_batch_norm1d(num_features=prod(deepest_shape)),
      activation_fn
    )

    # Convolutional network
    self$conv = torch::nn_sequential(
      upsample_block(channel_in=64, channel_out=32, act = activation_fn),
      upsample_block(channel_in=32, channel_out=16, act = activation_fn),
      torch::nn_conv2d(in_channels=16, out_channels=1, kernel_size=1, bias = FALSE)
    )
  },
  forward = function(x) {

    h <- x %>%
      self$fc()
    h <- h$view(c(x$shape[1], self$deepest_shape))
    self$conv(h)
  }
)

lsa <- torch::nn_module(
  initialize = function(input_shape, code_length, cpd_channels) {
    self$input_shape = input_shape
    self$code_length = code_length
    self$cpd_channels = cpd_channels

    # Build encoder
    self$encoder = encoder(
      input_shape=input_shape,
      code_length=code_length
    )

    # Build decoder
    self$decoder = decoder(
      code_length=code_length,
      deepest_shape=self$encoder$deepest_shape,
      output_shape=input_shape
    )

  },
  forward = function(x) {
    # Produce representations
    z = self$encoder(x)

    # Estimate CPDs with autoregression
    #z_dist = self.estimator(z)

    # Reconstruct x
    x_r = self$decoder(z)

    list(x_r = x_r, z = z)
  }
)
