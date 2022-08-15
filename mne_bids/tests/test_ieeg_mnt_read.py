from pathlib import Path
import mne_bids


def test_read_ieeg_coord_frame():
"""Ensure that the iEEG coordinate frame is read correctly."""
    bids_path = mne_bids.BIDSPath(
        subject="01",
        session="01",
        task="audiovisual",
        run="01",
        root=Path(__file__).parent / "data" / "ieeg_bids",
    )
    raw_read = mne_bids.read_raw_bids(bids_path)

    coord_frame = raw_read.get_montage().get_positions()["coord_frame"]
    assert coord_frame == "MNI152NLin2009bAsym"


if __name__ == '__main__':
    test_read_ieeg_bids_data()
