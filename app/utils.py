import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def tflog2pandas(path):
    """
    Convert TensorFlow log files to pandas DataFrame.

    Parameters
    ----------
    path : str
        Path to the TensorFlow log directory.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the scalar metrics.
    """

    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        data_ = []
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = [x.value for x in event_list]
            step = [x.step for x in event_list]
            data = pd.DataFrame(
                {
                    "metric": [tag.replace("Avg. cost per period/", "")] * len(step),
                    "value": values,
                    "step": step,
                }
            )
            data_.append(data)
        data = pd.concat(data_)
        data = data.loc[data["step"] >= 100]
    except Exception:
        print(f"Event file possibly corrupt: {path}")
        return pd.DataFrame()
    return data
