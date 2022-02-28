from pathlib import Path
import mne_bids


def test_read_ieeg_bids_data():

    bids_path = mne_bids.BIDSPath(
        subject="01",
        session="01",
        task="audiovisual",
        run="01",
        root=Path(__file__).parent / "data" / "ieeg_bids",
    )

    raw_read = mne_bids.read_raw_bids(bids_path)

    assert (
        raw_read.get_montage().get_positions()["coord_frame"] == "MNI152NLin2009bAsym"
    )


if __name__ == '__main__':
    test_read_ieeg_bids_data()
