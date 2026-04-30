cd checkpoints/stage1_vqvae

# lat16 → cpr8_img256
for d in */; do
    new=$(echo "$d" | sed 's/lat16/cpr8_img256/g' | sed 's/lat32/cpr4_img256/g' | sed 's/lat64/cpr2_img256/g')
    if [ "$d" != "$new" ]; then
        mv "$d" "$new"
        echo "renamed: $d → $new"
    fi
done