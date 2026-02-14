import flask
import fasttext



website = flask.Flask('API')
ft_model = fasttext.load_model('data/cc.en.300.bin')

@website.route('/')

def heartbeat():
    return flask.jsonify({'alive': True})
