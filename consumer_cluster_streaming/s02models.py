from tensorflow.python.keras.models import load_model

from functions.progress import ProgressShower
from consumer_cluster_streaming.common import get_xy_from_css, CCSManager
from consumer_cluster_streaming.lstm import utils
from consumer_cluster_streaming.lstm.tax import train_new_tax_model, train_tax_model


def do(ccs_manager: CCSManager):
    """
    Clusters the consumers using the linear fit clustering features

    Parameters
    ----------
    ccs_manager: CCSManager
        Manager to apply on
    """

    # parameters
    ta = ccs_manager.settings.ta
    tb = ccs_manager.settings.tb
    for t in ProgressShower(ccs_manager.timestamps[:-1], pre=f'learning models [{ccs_manager.name}]'):
        fn_out = ccs_manager.fn_model(t)
        if fn_out.exists():
            continue

        x, y = get_xy_from_css(ccs_manager, t)

        training_sequence, validation_sequence = utils.xy_to_generators(x, y, min(x.shape[0], 1000), random_state=0)
        print(t)
        if t == ta + tb or not ccs_manager.settings.incremental_training:
            model = train_new_tax_model(n_lab=1, fn_temp=None,
                                        training_sequence=training_sequence,
                                        validation_sequence=validation_sequence)
        else:
            model = train_tax_model(load_model(ccs_manager.fn_model(t - 1)), training_sequence=training_sequence,
                                    validation_sequence=validation_sequence, fn_temp=None)

        model.save(ccs_manager.fn_model(t))


if __name__ == '__main__':
    do(CCSManager('test'))
