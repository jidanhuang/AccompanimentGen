from frechet_audio_distance import FrechetAudioDistance
# Specify the paths to your saved embeddings
background_embds_path = "/path/to/saved/background/embeddings.npy"
eval_embds_path = "/path/to/saved/eval/embeddings.npy"

# Compute FAD score while reusing the saved embeddings (or saving new ones if paths are provided and embeddings don't exist yet)
fad_score = frechet.score(
    "/path/to/background/set",
    "/path/to/eval/set",
    background_embds_path=background_embds_path,
    eval_embds_path=eval_embds_path,
    dtype="float32"
)