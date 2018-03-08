#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "HayesTsai"

import os


class InputCorpus(object):
    def __init__(self, m_dir, encoding, suffix_accepted='txt,TXT'):
        if os.path.isdir(m_dir):
            self.input_path = m_dir
            self.encoding = encoding
        else:
            raise Exception(str(m_dir) + " does not exist!")
        self.input_files = []
        self.filenames = []
        self.targets = []
        self.suffix_accepted = tuple(suffix_accepted.split(','))

    def get_files(self):
        if not self.input_files:
            ret_files = []
            for root, _, files in os.walk(self.input_path, topdown=False):
                for name in files:
                    if name[0] == '.' or not name.endswith(self.suffix_accepted):
                        pass
                    if '-' in name:
                        clazz = name.split('-')[0][1:]
                    else:
                        clazz = -1
                    file_path = os.path.abspath(os.path.join(root, name))
                    ret_files.append(
                        InputFile(
                            file_path,
                            clazz,
                            from_encoding=self.encoding,
                            to_encoding='utf-8'
                        )
                    )
            self.input_files = ret_files
        return self.input_files

    def get_filenames_and_targets(self):
        if not self.input_files:
            self.get_files()
        if self.filenames == [] or self.targets == []:
            self.filenames = []
            self.targets = []
            for input_file in self.input_files:
                self.filenames.append(input_file.get_name())
                self.targets.append(input_file.get_class())
        return self.filenames, self.targets


class InputFile(object):
    def __init__(self, file_path, clazz, from_encoding, to_encoding):
        self.path = file_path
        self.encoding = to_encoding
        self.clazz = clazz
        import codecs
        with codecs.open(file_path, encoding=from_encoding, errors='ignore') as m_f:
            self.content = "".join(m_f.readlines())

    def get_content(self):
        return self.content

    def get_path(self):
        return self.path

    def get_name(self):
        return self.path.split('/')[-1]

    def get_class(self):
        return self.clazz


if __name__ == '__main__':
    input_corpus = InputCorpus('../corpus', 'gb18030')
    files = input_corpus.get_files()
    print(files[0].get_content())
