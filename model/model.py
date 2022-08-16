import torch.nn as nn


class GlobalPointer(nn.Module):
    """全局指针模块
    将序列的每个(start, end)作为整体来进行判断
    参考：https://kexue.fm/archives/8373
    """

    def __init__(
            self,
            hidden_size,
            heads,
            head_size,
            RoPE=True,
            use_bias=True,
            tril_mask=True,
            kernel_initializer='lecun_normal',

    ):
        super(GlobalPointer, self).__init__()
        self.heads = heads
        self.head_size = head_size
        self.RoPE = RoPE
        self.use_bias = use_bias
        self.tril_mask = tril_mask
        self.kernel_initializer = None  # initializers.get(kernel_initializer)
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, self.head_size * self.heads * 2, bias=True)
        )

    def build(self, input_shape):
        super(GlobalPointer, self).build(input_shape)

    def forward(self, inputs, mask=None):
        # 输入变换
        inputs = self.dense(inputs)
        inputs = tf.split(inputs, self.heads, axis=-1)
        inputs = K.stack(inputs, axis=-2)
        qw, kw = inputs[..., :self.head_size], inputs[..., self.head_size:]
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            qw, kw = apply_rotary_position_embeddings(pos, qw, kw)
        # 计算内积
        logits = tf.einsum('bmhd,bnhd->bhmn', qw, kw)
        # 排除padding
        logits = sequence_masking(logits, mask, '-inf', 2)
        logits = sequence_masking(logits, mask, '-inf', 3)
        # 排除下三角
        if self.tril_mask:
            mask = tf.linalg.band_part(K.ones_like(logits), 0, -1)
            logits = logits - (1 - mask) * K.infinity()
        # scale返回
        return logits / self.head_size ** 0.5


class EfficientGlobalPointer(GlobalPointer):
    """更加参数高效的GlobalPointer
    参考：https://kexue.fm/archives/8877
    """

    def build(self, input_shape):
        self.p_dense = Dense(
            units=self.head_size * 2,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.q_dense = Dense(
            units=self.heads * 2,
            use_bias=self.use_bias,
            kernel_initializer=self.kernel_initializer
        )
        self.built = True

    def forward(self, inputs, mask=None):
        # 输入变换
        inputs = self.p_dense(inputs)
        qw, kw = inputs[..., ::2], inputs[..., 1::2]
        # RoPE编码
        if self.RoPE:
            pos = SinusoidalPositionEmbedding(self.head_size, 'zero')(inputs)
            qw, kw = apply_rotary_position_embeddings(pos, qw, kw)
        # 计算内积
        logits = tf.einsum('bmd,bnd->bmn', qw, kw) / self.head_size ** 0.5
        bias = tf.einsum('bnh->bhn', self.q_dense(inputs)) / 2
        logits = logits[:, None] + bias[:, ::2, None] + bias[:, 1::2, :, None]
        # 排除padding
        logits = sequence_masking(logits, mask, '-inf', 2)
        logits = sequence_masking(logits, mask, '-inf', 3)
        # 排除下三角
        if self.tril_mask:
            mask = tf.linalg.band_part(K.ones_like(logits), 0, -1)
            logits = logits - (1 - mask) * K.infinity()
        # 返回最终结果
        return logits


class GPLinker(nn.Module):
    def __init__(self, encoder, predicate2id):
        super(GPLinker, self).__init__()
        self.entity_output = GlobalPointer(hidden_size=encoder.config.hidden_size, heads=2, head_size=64)
        self.head_output = GlobalPointer(
            hidden_size=encoder.config.hidden_size, heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
        )
        self.tail_output = GlobalPointer(
            hidden_size=encoder.config.hidden_size, heads=len(predicate2id), head_size=64, RoPE=False, tril_mask=False
        )

    def forward(self, x):
        return x
