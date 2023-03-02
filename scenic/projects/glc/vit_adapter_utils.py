from ml_collections import ConfigDict, FrozenConfigDict

def get_vit_filename(query):
    df = checkpoint.get_augreg_df()
    res = df.query(query).filename.unique()
    assert len(res) == 1
    return res[0]

def get_vit_checkpoint(image_size, query):
    filename = get_vit_filename(query)

    config = get_vit_config(query)

    model = VisionTransformer(**config, num_classes=2)  # num_classes unused.
    init_params = model.init(jax.random.PRNGKey(0),
                           get_sample_image(image_size=image_size,
                                            batch_size=1),
                           train=USE_DROPOUT)['params']

    params = checkpoint.load_pretrained(
    pretrained_path=f'gs://vit_models/augreg/{filename}.npz',
    init_params=init_params,
    model_config=config)

    return params

def get_vit_checkpoint_mapped(image_size, query):
    params = get_vit_checkpoint(image_size, query)
    params = params_model_to_comps(params)
    return params

# Parameter mapping.
TRANSFORMER_KEYS = set()
# Set this to True to unfreeze all the layernorms in the model.
# Can be useful for variants of the residual adapters baseline.
GATHER_LAYER_NORMS = False

def params_model_to_comps(params):
    global TRANSFORMER_KEYS
    TRANSFORMER_KEYS.update(params['Transformer'].keys())
    new_params = {}
    for k in params.keys():
    if k == 'Transformer':
        t_params = params[k]
        for t_k in t_params.keys():
            new_params[t_k] = t_params[t_k]
    else:
        new_params[k] = params[k]
    params = flax.core.freeze(new_params)

    if GATHER_LAYER_NORMS:
    params = params.unfreeze()
    params['encoder_norm']['gathered'] = {}
    for k in params.keys():
        if k.startswith('encoderblock_'):
            params['encoder_norm']['gathered'][k] = {}
            encoderblock_keys = list(params[k].keys())
            for ek in encoderblock_keys:
                if ek.startswith('LayerNorm_'):
                    params['encoder_norm']['gathered'][k][ek] = params[k].pop(ek)

    return flax.core.freeze(params)

def params_comps_to_model(params):
    params = params.unfreeze()

    if GATHER_LAYER_NORMS:
    gathered = params['encoder_norm'].pop('gathered')
    for k in gathered:
        assert k.startswith('encoderblock_')
        assert k in params
        for ke in gathered[k].keys():
            assert ke.startswith('LayerNorm_')
            assert ke not in params[k]
            params[k][ke] = gathered[k][ke]

    params['Transformer'] = {}
    keys = list(params.keys())
    assert len(TRANSFORMER_KEYS) != 0
    for k in keys:
        if k in TRANSFORMER_KEYS:
            params['Transformer'][k] = params.pop(k)
    return flax.core.freeze(params)


def format_params(a, b):
    params = a.copy(b)
    assert len(params) == len(a) + len(b)  # Dicts should not overlap.
    params = params_comps_to_model(params)
    return params