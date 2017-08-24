# created on 2017-8-22
# by xdliu


# strategy from "advanced algorithm trading arima & garch"
library(rugarch)
library(tseries)
library(lattice)

signal <- function (rt, window_length){
  
  forecast
}

best_arima <- function(x, max_p=5, max_i = 0, max_q=5){
  final.aic <- Inf
  final.order <- c(0, 0, 0)
  # final.arima <- arima(x, order=c(1,0,1))
  for (p in 0:max_p) for (i in 0:max_i) for (q in 0:max_q){
    fit = tryCatch(arima(x, order=c(p, i, q)),
                   error=function(err) FALSE,
                   warning=function(err) FALSE)
    if (!is.logical(fit)){
      current_aic <- AIC(arima(x, order=c(p, i, q)))
      if (current_aic < final.aic){
        final.aic <- current_aic
        final.order <- c(p, i, q)
        final.arima <- arima(x, order=final.order)
      }
    } else{
      next
    }
    
  }
  # final.order <- list(final.order)
  # names(final.order) <- c('p', 'i', 'q')
  result <- list(final.aic, final.order, final.arima)
  names(result) <- c('final.aic', 'final.order', 'final.arima')
  return(result)
}

model_garch <-function(x, garchOrder=c(1,1), armaOrder=c(1,1)){
  spec <- ugarchspec(
    variance.model = list(garchOrder=garchOrder),
    mean.model = list(armaOrder=armaOrder, include.mean = T),
    distribution.model = "sged"
  )
  fit = tryCatch(
    ugarchfit(spec, x, solver = 'hybrid'),
    error=function(e) FALSE,
    warning=function(e) FALSE
  )
  return(fit)
}

fit_and_predict <- function(x, p=1, i=0, q=1, garchOrder=c(1,1)){
  param <- best_arima(x, p, i, q)
  #armaOrder <- c(param$final.order[1], param$final.order[3])
  model <- model_garch(x, garchOrder=c(1,1),
                       armaOrder=c(param$final.order[1], param$final.order[3]))
  if (!is.logical(model)){
    forecast_model <- tryCatch(ugarchforecast(model, n.ahead = 1),
                              error=function(e) FALSE,
                              warning=function(e) FALSE)
    if (is.logical(forecast_model)){
      predict_value <- 1
    } else{
      predict_value <- forecast_model@forecast$seriesFor[1]
    }
  } else{
    predict_value <- 1
  }
  return(predict_value)
}

#df <- read.csv('600284SH.csv')
#df <- read.csv('FG0.csv')
#rt = diff(log(df$Close))
#for (i in 1:100){
#  slice <- rt[i:(i+199)]
#  #print(slice)
#  # param <- best_arima(rt, max_p = 5, max_i = 0, max_q = 5)
#  result <- fit_and_predict(slice, p=5, i=0, q=5)
#  print(result)
#}

