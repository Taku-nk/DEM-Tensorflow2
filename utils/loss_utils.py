import pandas as pd

def loss_history_to_csv(loss_history, save_path=''):
    """ Loss history to csv.
    Convert loss history list of dictionary to csv data and save it.
    Args:
        loss_history: list. List of dictionary which contains 'i': iter, 'loss': loss, 'id': 1 (0: adam, 1: lbfgs)
        save_path: path_like. csv file will be save to this specified path (must inlcude file name)
    Returns:
        -
    """
    loss_df = pd.DataFrame.from_dict(loss_history)
    loss_df.to_csv(save_path, index=False)
