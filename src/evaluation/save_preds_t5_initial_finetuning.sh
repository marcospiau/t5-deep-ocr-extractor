# The argument on the input is the partition ('train' or 'test')
predictions_basedir=/home/marcospiau/final_project_ia376j/data/sroie_receipt_dataset/predictions
for i in 27 28 29 30 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48
do
    python3 save_experiment_predictions_t5_ocr_baseline.py \
        --neptune_experiment_id="FIN-$i" \
        --input_path="/home/marcospiau/final_project_ia376j/data/sroie_receipt_dataset/$1/" \
        --output_path="$predictions_basedir/FIN-$i/$1"
done
