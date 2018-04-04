from config import *
from weighted_vector import generate_feature_for_sentence, read_w2v
from basic_util import token_strip
file_w2v = W2V_PATH + "/" + W2V_FILE

w2v, dim = read_w2v(file_w2v)
print('done')
print('One:')
generate_feature_for_sentence('One', 4.5, 26, w2v, dim)
print('one:')
generate_feature_for_sentence('one', 4.5, 26, w2v, dim)
print("first one:")
generate_feature_for_sentence('first one', 4.5, 26, w2v, dim)
print("less than:")
generate_feature_for_sentence('less than', 4.5, 26, w2v, dim)