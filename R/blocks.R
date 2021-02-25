impl_residual_block <- torch::nn_module(
  initialize = function(conv1a, conv2a = identity, conv1b, bn1a, bn2a = identity,
                        bn1b, act) {
    self$conv1a <- conv1a
    self$conv2a <- conv2a
    self$conv1b <- conv1b
    self$bn1a   <- bn1a
    self$bn2a   <- bn2a
    self$bn1b   <- bn1b
    self$act    <- act
  },
  forward = function(x) {
    h1 <- x %>%
      self$conv1a() %>%
      self$bn1a() %>%
      self$act() %>%
      self$conv1b() %>%
      self$bn1b()

    h2 <- x %>%
      self$conv2a() %>%
      self$bn2a()

    # Residual connection
    out <- h1 + h2

    self$act(out)
  }
)

downsample_block <- torch::nn_module(
  inherit = impl_residual_block,
  initialize = function(channel_in, channel_out, act, use_bn=TRUE,
                        use_bias=FALSE) {

    conv2d <- purrr::partial(
      torch::nn_conv2d,
      out_channels=channel_out,
      bias=use_bias
    )

    if (use_bn) {
      batch_norm_2d <- purrr::partial(
        torch::nn_batch_norm2d,
        num_features=channel_out
      )
    } else {
      batch_norm_2d <- identity
    }

    super$initialize(
      conv1a = conv2d(channel_in, kernel_size=3, padding=1, stride=2),
      conv1b = conv2d(channel_out, kernel_size=3, padding=1, stride=1),
      conv2a = conv2d(channel_in, kernel_size=1, padding=0, stride=2),
      bn1a = batch_norm_2d(),
      bn1b = batch_norm_2d(),
      bn2a = batch_norm_2d(),
      act = act
    )

  }
)

upsample_block <- torch::nn_module(
  inherit = impl_residual_block,
  initialize = function(channel_in, channel_out, act, use_bn=TRUE, use_bias=FALSE) {

    conv_transpose_2d <- purrr::partial(
      torch::nn_conv_transpose2d,
      in_channels=channel_in,
      out_channels=channel_out,
      bias=use_bias
    )

    conv2d <- purrr::partial(
      torch::nn_conv2d,
      in_channels=channel_out,
      out_channels=channel_out,
      bias=use_bias
    )

    if (use_bn) {
      batch_norm_2d <- purrr::partial(
        torch::nn_batch_norm2d,
        num_features=channel_out
      )
    } else {
      batch_norm_2d <- identity
    }

    super$initialize(
      conv1a = conv_transpose_2d(
        kernel_size=5,
        padding=2,
        stride=2,
        output_padding=1
      ),
      conv1b = conv2d(kernel_size=3, padding=1, stride=1),
      conv2a = conv_transpose_2d(
        kernel_size=1,
        padding=0,
        stride=2,
        output_padding=1
      ),
      bn1a = batch_norm_2d(),
      bn1b = batch_norm_2d(),
      bn2a = batch_norm_2d(),
      act = act
    )
  }
)

residual_block <- torch::nn_module(
  inherit = impl_residual_block,
  initialize = function(channel_in, channel_out, act, use_bn=TRUE, use_bias=FALSE) {

    conv2d <- purrr::partial(
      torch::nn_conv2d,
      in_channels=channel_in,
      out_channels=channel_out,
      bias=use_bias
    )

    if (use_bn) {
      batch_norm_2d <- purrr::partial(
        torch::nn_batch_norm2d,
        num_features=channel_out
      )
    } else {
      batch_norm_2d <- identity
    }

    super$initialize(
      conv1a = conv2d(kernel_size=3, padding=1, stride=1),
      conv1b = conv2d(kernel_size=3, padding=1, stride=1),
      bn1a = batch_norm_2d(),
      bn1b = batch_norm_2d(),
      act = act
    )

  }
)
