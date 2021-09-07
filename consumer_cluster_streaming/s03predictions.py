import pandas as pd
from tensorflow.python.keras.models import load_model

from consumer_cluster_streaming import common as sc
from functions.dataframe_operations import export_df
from functions.progress import ProgressShower
from consumer_cluster_streaming.common import get_xy_from_css, CCSManager


def do(ccs_manager: CCSManager):
    for t in ProgressShower(ccs_manager.timestamps[:-1], pre=f'making predictions [{ccs_manager.name}]'):
        fn_out = ccs_manager.fn_cluster_predictions(t)
        if fn_out.exists():
            continue

        x_test, turnover_per_cluster_true = get_xy_from_css(ccs_manager, t + ccs_manager.settings.tb, t)
        model = load_model(ccs_manager.fn_model(t))
        turnover_per_cluster_pred = model.predict(x_test)
        export_df(df=pd.DataFrame(index=pd.RangeIndex(stop=len(x_test), name=sc.CLUSTER),
                                  data={'y_pred': turnover_per_cluster_pred.flatten(),
                                        'y_true': turnover_per_cluster_true}),
                  fn=fn_out,
                  index=True)
