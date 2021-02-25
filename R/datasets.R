#' One class mnist dataset
#'
#' Filter a single class of the mnist dataset for training
#'
#' @param class the digit.
#' @param split 'train', 'valid' or 'test'.
#' @param valid_split the fraction of the full training set used for validation.
#'
#' @export
one_class_mnist <- torch::dataset(
  "mnist",
  inherit = torchvision::mnist_dataset,
  initialize = function(class, split = "train", valid_split = 0.1, ...) {

    super$initialize(..., train = split != "test")

    if (split != "test") {
      set.seed(121)
      n <- length(self$targets)
      valid_indices <- sample.int(n, size = as.integer(valid_split*n))

      if (split == "train") {
        indices <- which(self$targets == class + 1)
        indices <- indices[!indices %in% valid_indices]
      } else if (split == "valid") {
        indices <- valid_indices
      }


      self$targets <- self$targets[indices]
      self$data <- self$data[indices,,,drop=FALSE]
    }
  }
)

