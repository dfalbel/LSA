test_that("lsa", {

  input <- torch::torch_randn(32, 1, 28, 28)
  model <- lsa(c(1, 28, 28), 50, 1)
  out <- model(input)

  expect_equal(out$x_r$shape, c(32, 1, 28, 28))
  expect_equal(out$z$shape, c(32, 50))

})
