for seed in 1 2 3 4 5  ; do
python train/jax_sampler_ppo.py --sampler gmmvi --gmm-target-beta=1 --seed=$seed
done
for seed in 1 2 3 4 5; do
python train/jax_sampler_ppo.py --sampler gmmvi --gmm-target-beta=-1 --seed=$seed
done

for seed in 1 2 3 4 5; do
python train/jax_sampler_ppo.py --sampler gmmvi --gmm-target-beta=-2  --seed=$seed
done
# for seed in 1 2 3 4 5; do
# python train/jax_sampler_ppo.py --sampler gmmvi --gmm-target-beta=-10  --seed=$seed
# done
# for seed in 1 2 3 4 5; do
# python train/jax_sampler_ppo.py --sampler gmmvi --gmm-target-beta=-20  --seed=$seed
# done

for seed in 1 2 3 4 5; do
python train/jax_sampler_ppo.py --seed=$seed
done
