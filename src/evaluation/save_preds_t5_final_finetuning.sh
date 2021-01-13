# My Dataset implementation is not prepared to be used with no labels, as in the final test.
# As a workaround, I created .txt files with empty predictions for all fields.
predictions_basedir=/home/marcospiau/final_project_ia376j/data/sroie_receipt_dataset/predictions
for i in 50 51 52 53
do
    python3 save_experiment_predictions_t5_ocr_baseline.py \
        --neptune_experiment_id="FIN-$i" \
        --input_path="/home/marcospiau/final_project_ia376j/data/sroie_receipt_dataset/final_competition_test/" \
        --output_path="$predictions_basedir/FIN-$i/"
done
