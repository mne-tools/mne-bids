from mne_bids.utils import _check_types, create_types, filename_prefix
from collections import OrderedDict
PREFIX_DATA = OrderedDict(subject='one', session='two', task='three',
                          acquisition='four', run='five', processing='six',
                          recording='seven', suffix='suffix.csv')

my_name = filename_bids(**PREFIX_DATA)
assert my_name == '_'.join('%s-%s' % (key, val) for key, val in PREFIX_DATA.items())

# make sure leaving out keys works
for key in PREFIX_DATA.keys():
    this_data = PREFIX_DATA.copy()
    this_data.pop(key)
    assert my_name == '_'.join('%s-%s' % (key, val) for key, val in this_data.items())
