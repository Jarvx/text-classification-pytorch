from torchtext import data
import os


class SST1Dataset(data.TabularDataset):
    dirname = 'data'
    @classmethod
    def splits(cls, text_field, label_field,
               train='phrases.train.tsv', validation='dev.tsv', test='test.tsv'):
        prefix_name = 'stsa.fine.'
        path = './data'
        return super(SST1Dataset, cls).splits(
            path, train=prefix_name + train, validation=prefix_name + validation, test=prefix_name + test,
            format='TSV', fields=[('label', label_field), ('text', text_field)]
        )


class SST2Dataset(data.TabularDataset):
    dirname = 'data/sst2'
    @classmethod
    def splits(cls, text_field, label_field,
               train='phrases.train.tsv', validation='dev.tsv', test='test.tsv'):
        prefix_name = 'stsa.binary.'
        path = './data/sst2'
        return super(SST2Dataset, cls).splits(
            path, train=prefix_name + train, validation=prefix_name + validation, test=prefix_name + test,
            format='TSV', fields=[('label', label_field), ('text', text_field)]
        )


class MRDataset(data.TabularDataset):
    dirname = 'data/mr'
    @classmethod
    def splits(cls, text_field, label_field,
               train='train.tsv', validation='dev.tsv'):
        prefix_name = 'rt-polarity.'
        path = './data/mr'
        return super(MRDataset, cls).splits(
            path, train=prefix_name + train, validation=prefix_name + validation,
            format='TSV', fields=[('label', label_field), ('text', text_field)]
        )
