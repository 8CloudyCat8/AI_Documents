from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")

sequence_to_classify = "Вараники со сметаной"
candidate_labels = ['спорт', 'еда', 'наука']

result = classifier(sequence_to_classify, candidate_labels)

highest_score_index = result['scores'].index(max(result['scores']))
highest_score_label = result['labels'][highest_score_index]
highest_score = result['scores'][highest_score_index]

print("Результаты классификации:")
print(result)
print(f"Категория с наивысшим баллом: '{highest_score_label}' с баллом {highest_score:.4f}")
