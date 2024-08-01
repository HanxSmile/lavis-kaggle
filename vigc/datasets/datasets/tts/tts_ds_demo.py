from torch.utils.data import Dataset


class TTSDataset(Dataset):

    def __init__(
            self,
            dataset_name,
            audio_column_name='audio',
            text_column_name='text',
            speaker_id_column_name=None,
            filter_on_speaker_id
    ):