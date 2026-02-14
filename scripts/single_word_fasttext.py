import fasttext

#code to import the fasttext library in: source: https://fasttext.cc/docs/en/python-module.html
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('cc.en.300.bin')

#embedding printed here
print(ft.get_word_vector("sleep"))