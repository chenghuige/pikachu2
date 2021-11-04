import oneflow as flow


def constant_initializer(val=0.1):
    return flow.constant_initializer(val)


def Variable(name, shape, initializer=None, dtype=flow.float, distribute=flow.distribute.auto()):
    if initializer is None:
        initializer = flow.glorot_uniform_initializer()
    return flow.get_variable(name=name, shape=shape, dtype=dtype, initializer=initializer, distribute=distribute)

class Wide(object):
    def __init__(self, config, model_distribute):
        self.feat_dict_size = config.feature_dict_size
        self.field_size = config.field_size
        self.max_feat_len = config.max_feat_len
        self.model_distribute_ = model_distribute


    def forward(self, features):
        feat_ids = features['index']
        feat_fields = features['field']
        feat_values = features['value']

        W = Variable('w', [self.feat_dict_size, 1], constant_initializer(0.01),
                     distribute=self.model_distribute_)
        W = W.with_distribute(self.model_distribute_)
        b = Variable('b', [1], constant_initializer(0))

        feat_weights = flow.gather(params=W, indices=feat_ids, axis=0)
        feat_values = flow.reshape(feat_values, [-1, self.max_feat_len, 1])
        wide_weighted_feats = feat_values * feat_weights

        #wide_field_sum = flow.math.segment_sum(
        wide_field_sum = flow.unsorted_batch_segment_sum(
            wide_weighted_feats, feat_fields, self.field_size)
        wide_field_sum = flow.reshape(wide_field_sum, [-1, self.field_size])
        wide_scores = flow.math.reduce_sum(wide_field_sum, axis=[1])
        wide_scores = wide_scores + b

        return wide_scores

class Deep(object):
    def __init__(self, config, model_distribute):
        self.feature_dict_size = config.feature_dict_size
        self.embedding_size = config.embedding_size
        self.deep_layers = config.deep_layers
        self.field_size = config.field_size
        self.model_distribute_ = model_distribute

        self.max_feat_len = config.max_feat_len
        # we combine a batch data to one piece data


    def forward(self, features):
        feat_ids = features['index']
        feat_fields = features['field']

        feat_ids = feat_ids % flow.constant(self.feature_dict_size, dtype=feat_ids.dtype)

        feat_embedding_table = Variable('feat_embeddings',
                                        [self.feature_dict_size, self.embedding_size],
                                        distribute=self.model_distribute_)
        feat_embedding_table = feat_embedding_table.with_distribute(self.model_distribute_)
        feat_embeddings = flow.gather(feat_embedding_table, feat_ids, axis=0)

        #deep_field_sum = flow.math.segment_sum(
        deep_field_sum = flow.unsorted_batch_segment_sum(
            feat_embeddings, feat_fields, self.field_size)
        deep_field_concat = flow.reshape(deep_field_sum, [-1, self.field_size*self.embedding_size])

        deep_h_o = deep_field_concat
        for i in range(1, len(self.deep_layers)):
            deep_w_h_i = Variable('w_h_%d'%i, [self.deep_layers[i-1], self.deep_layers[i]])
            deep_b_h_i = Variable('b_h_%d'%i, [self.deep_layers[i]], constant_initializer(0))
            deep_h_o = flow.matmul(deep_h_o, deep_w_h_i)
            deep_h_o = deep_h_o + deep_b_h_i
            deep_h_o = flow.nn.relu(deep_h_o)

        deep_w_o = Variable('w_o', [self.deep_layers[-1], 1])
        deep_b_o = Variable('b_o', [1], constant_initializer(0))

        deep_scores = flow.matmul(deep_h_o, deep_w_o)
        deep_scores = flow.reshape(deep_scores, [-1, ])
        deep_scores = deep_scores + deep_b_o

        return deep_scores


def WideAndDeep(config, features, labels, model_distribute = flow.distribute.auto()):
    wide = Wide(config, model_distribute)
    wide_scores = wide.forward(features)

    deep = Deep(config, model_distribute)
    deep_scores = deep.forward(features)

    scores = wide_scores + deep_scores

    scores = flow.reshape(scores, [-1, 1])
    sigmoid_loss = flow.nn.sigmoid_cross_entropy_with_logits(logits=scores, labels=labels)
    cost = flow.math.reduce_sum(sigmoid_loss, name="sigmoid_loss")
    predict = flow.math.sigmoid(scores)
    return cost, predict, labels
