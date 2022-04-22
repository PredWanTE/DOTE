from tensorflow.keras import callbacks
from ml.sl_algos.nn import metrics
from tensorflow.keras import optimizers


def compile_model(model, props):
    loss_ = 'mse'
    if props.use_imitation:
        loss_ = metrics.mean_square_percent,

    model.compile(loss=loss_,
                  optimizer=optimizers.Adam())


def get_extra_nn(props, X_test, Y_test, use_callbacks=None):
    if use_callbacks is None:
        use_callbacks = [callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min')]

    if props.early_stopping:
        use_callbacks += [
            callbacks.EarlyStopping(patience=props.early_stopping)
        ]

    if props.dir_name:
        use_callbacks += [
            callbacks.CSVLogger(get_dir(props) + '/training.log'),
            callbacks.ModelCheckpoint(get_dir(props) + "/model.{epoch:02d}-{val_loss:.2f}.hdf5",
                                      period=props.dump_policy_every),

        ]

    res_dict = {'epochs': props.num_epochs,
                'verbose': 2,
                'batch_size': props.sl_batch_size,
                'validation_data': (X_test, Y_test),
                'callbacks': use_callbacks
                }
    return res_dict


def get_dir(props):
    dir_name = props.dir_name.strip() + "/" + props.sl_model_type.strip()
    import os
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


def dump_model(props, model, model_path=None):
    model_path = model_path or "%s/model" % get_dir(props)
    model.save(model_path)


def dump_history(props, history, history_path=None):
    history_path = history_path or props.dir_name + "/" + props.sl_model_type + "/history"
    import dill
    with open(history_path, 'wb') as f:
        dill.dump(history, f)
