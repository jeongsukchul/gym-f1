
for seed in 0 1 2 3 4; do
python train/jax_sampler_ppo.py --sampler gmmvi --seed=$seed
done

for seed in 0 1 2 3 4; do
python train/jax_sampler_ppo.py --seed=$seed
done
