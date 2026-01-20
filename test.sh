declare -a dataset=(MVTec MPDD BTAD Colon_clinicDB Colon_colonDB Colon_Kvasir Colon_cvc300 Liver Brain Retina)
save_path="./ckpt/test"

for i in "${dataset[@]}"; do
    python testUP.py --save_path $save_path --dataset $i 
done