from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR as SVMRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import BayesianRidge
from statsmodels.tsa.arima.model import ARIMA

from statsmodels.tsa.arima.model import ARIMA as Arima
from sklearn.linear_model import Ridge

LINEAR_REGRESSION = "linear_regression"
POLYNOMIAL_REGRESSION = "polynomial_regression"
LOGISTIC_REGRESSION = "logistic_regression"
RIDGE_REGRESSION = "ridge_regression"
BASYIAN_INFERENCE = "basyian_inference"
RANDOM_FOREST = "random_forest"
ARIMA = "arima"
ARMA = "arma"
SVR = "svr"

ALL_MODELS = [BASYIAN_INFERENCE, POLYNOMIAL_REGRESSION, LINEAR_REGRESSION, RIDGE_REGRESSION]


def linear_regression(env,props):
    return LinearRegression(n_jobs=-1, fit_intercept=False)


def logistic_regression(env,props):
    return LogisticRegression(n_jobs=-1, fit_intercept=False)


def basyian_inference(env,props):
    return BayesianRidge(fit_intercept=False)


def polynomial_regression(env,props):
    estimators = []
    estimators.append(('poly', PolynomialFeatures(degree=2)))
    estimators.append(('pred', ridge_regression(env,props)))
    model = Pipeline(estimators)
    return model


def ridge_regression(env,props):
    return Ridge(fit_intercept=False)


def svr(env,props):
    return SVMRegression()


def arima(env, props):
    return None


def random_forest(env, props):
    return RandomForestRegressor(n_estimators=100)


def get_model(env, props, force_model = None):
    if force_model:
        props.sl_model_type = force_model
        
    if props.sl_model_type == LINEAR_REGRESSION:
        model = linear_regression(env, props)
    
    elif props.sl_model_type == LOGISTIC_REGRESSION:
        model = logistic_regression(env, props)
    
    elif props.sl_model_type == RIDGE_REGRESSION:
        model = ridge_regression(env, props)
    
    elif props.sl_model_type == SVR:
        model = svr(env, props)
    
    elif props.sl_model_type == POLYNOMIAL_REGRESSION:
        model = polynomial_regression(env, props)
 
    elif props.sl_model_type == BASYIAN_INFERENCE:
        model = basyian_inference(env, props)

    elif props.sl_model_type == ARIMA:
        model = arima(env, props)

    elif props.sl_model_type == RANDOM_FOREST:
        model = random_forest(env, props)

    return model
