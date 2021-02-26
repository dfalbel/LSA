masked_fully_connection <- torch::nn_module(
  inherit = torch::nn_linear,
  initialize = function(mask_type, in_channels, out_channels, ...) {
    self$mask_type = mask_type
    self$in_channels = in_channels
    self$out_channels = out_channels

    super$initialize(...)
    diagonal <- if (mask_type == "A") -1 else 0

    mask <-  self$weight %>%
      torch::torch_ones_like(requires_grad = FALSE) %>%
      torch:::torch_tril(diagonal)

    self$register_buffer('mask', mask)
  },
  forward = function(x) {
    # Reshape
    x <- torch::torch_transpose(x, 2, 3)$contiguous()
    x <- x$view(c(x$shape[1], -1))

    # Mask weights and call fully connection
    torch::with_no_grad({
      self$weight$mul_(self$mask)
    })

    o <- super$forward(x)

    # Reshape again
      o <- o$view(c(o$shape[1], -1, self$out_channels))
    torch::torch_transpose(o, 2, 3)$contiguous()
  }
)

estimator <- torch::nn_module(
  initialize = function(code_length, fm_list, cpd_channels) {

    self$code_length = code_length
    self$fm_list = fm_list
    self$cpd_channels = cpd_channels

    activation_fn = torch::nn_leaky_relu()

    mask_types <- c("A", rep("B", length(fm_list)))
    fm_ins <- c(1, fm_list)
    fm_outs <- c(fm_list, cpd_channels)
    activations <- c(purrr::rerun(length(fm_list), activation_fn), torch::nn_identity())

    layers <- purrr::pmap(
      list(mask_types, fm_ins, fm_outs, activations),
      function(mask_type, fm_in, fm_out, activation_fn) {
        torch::nn_sequential(
          masked_fully_connection(
            mask_type=mask_type,
            in_features=fm_in * code_length,
            out_features=fm_out * code_length,
            in_channels=fm_in,
            out_channels=fm_out
          ),
          activation_fn
        )
      }
    )

    self$layers <- torch::nn_sequential(!!!layers)

  },
  forward = function(x) {
    h = torch::torch_unsqueeze(x, dim=2)  # add singleton channel dim
    h = self$layers(h)
    h
  }
)
