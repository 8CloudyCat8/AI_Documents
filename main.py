import sys
import asyncio
import json
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QTextEdit, QLineEdit, QLabel, QProgressBar, QSlider, QHBoxLayout,
    QFileDialog, QCheckBox, QFormLayout, QTabWidget, QGroupBox, QListWidget, QListWidgetItem, QMessageBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from docx import Document
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
from rutermextract import TermExtractor
import os

zero_shot_pipeline = pipeline(
    task="zero-shot-classification",
    model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)

term_extractor = TermExtractor()

class AnalysisThread(QThread):
    progress = pyqtSignal(int)
    new_result = pyqtSignal(str)
    result_ready = pyqtSignal(list)
    analysis_finished = pyqtSignal(dict)

    def __init__(self, files_data, threshold):
        super().__init__()
        self.files_data = files_data
        self.threshold = threshold

    def classify_text(self, sequence, theme):
        result = zero_shot_pipeline(
            sequences=sequence,
            hypothesis_template="Этот текст о {}.",
            candidate_labels=[theme]
        )
        return result

    async def run_in_executor(self, sequences, theme):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            tasks = [
                loop.run_in_executor(pool, self.classify_text, seq, theme)
                for seq in sequences
            ]
            total_tasks = len(tasks)
            results = []

            output_file = 'analysis_results.json'

            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = {}
            else:
                existing_data = {}

            if theme not in existing_data:
                existing_data[theme] = []

            for idx, task in enumerate(asyncio.as_completed(tasks)):
                result = await task
                self.progress.emit(int((idx + 1) / total_tasks * 100))
                results.append(result)

                sorted_results = sorted(results, key=lambda x: x['scores'][0], reverse=True)

                for result in sorted_results:
                    if result['labels'][0] == theme and result['scores'][0] >= self.threshold:
                        matching_phrase = (result['sequence'], result['scores'][0])
                        existing_phrases = {phrase[0] for phrase in existing_data[theme]}

                        if matching_phrase[0] not in existing_phrases:
                            existing_data[theme].append(matching_phrase)

                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(existing_data, f, ensure_ascii=False, indent=4)

                            self.new_result.emit(
                                f"'{matching_phrase[0]}', Сходство с темой: {matching_phrase[1] * 100:.2f}%"
                            )

            return results

    async def extract_and_classify(self, text, theme):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as pool:
            term_sequences = [term.normalized for term in term_extractor(text)]
            results = await self.run_in_executor(term_sequences, theme)
            filtered_results = [
                (result['sequence'], result['scores'][0])
                for result in results
                if result['labels'][0] == theme and result['scores'][0] >= self.threshold
            ]
            return filtered_results

    def run(self):
        all_results = {}
        for file_path, theme in self.files_data:
            document = Document(file_path)
            text = '\n'.join([paragraph.text for paragraph in document.paragraphs])
            matching_phrases = asyncio.run(self.extract_and_classify(text, theme))
            all_results[theme] = matching_phrases

        self.analysis_finished.emit(all_results)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

        with open('style.qss', 'r', encoding='utf-8') as f:
            stylesheet = f.read()
        self.setStyleSheet(stylesheet)

        self.load_keywords()

    def init_ui(self):
        self.setWindowTitle('Анализ текста с классификацией')
        self.resize(600, 600)

        layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.tab_analysis = QWidget()
        self.tab_keywords = QWidget()

        self.tabs.addTab(self.tab_analysis, "Анализ")
        self.tabs.addTab(self.tab_keywords, "Ключевые фразы")

        self.init_analysis_tab()

        self.init_keywords_tab()

        layout.addWidget(self.tabs)
        self.setLayout(layout)
        self.show()

    def init_keywords_tab(self):
        self.keywords_layout = QVBoxLayout()
        self.tab_keywords.setLayout(self.keywords_layout)

    def load_keywords(self):
        output_file = 'analysis_results.json'

        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = {}

            for i in reversed(range(self.keywords_layout.count())):
                widget = self.keywords_layout.itemAt(i).widget()
                if widget is not None:
                    widget.deleteLater()

            if data:
                for theme, phrases in data.items():
                    self.create_keyword_group(theme, phrases)
            else:
                self.keywords_layout.addWidget(QLabel("Нет сохраненных ключевых фраз."))
        else:
            self.keywords_layout.addWidget(QLabel("Файл с ключевыми фразами не найден."))

    def create_keyword_group(self, theme, phrases):
        group_box = QGroupBox(f"Тема: {theme}")
        layout = QVBoxLayout()

        list_widget = QListWidget()

        for phrase, score in phrases:
            list_item = QListWidgetItem(f"{phrase}, {score:.4f}%")
            list_widget.addItem(list_item)

        layout.addWidget(list_widget)

        delete_button = QPushButton('Удалить тему', self)
        delete_button.clicked.connect(lambda: self.delete_theme(theme))
        layout.addWidget(delete_button, alignment=Qt.AlignRight)

        group_box.setLayout(layout)
        self.keywords_layout.addWidget(group_box)

    def delete_theme(self, theme):
        reply = QMessageBox.question(
            self,
            'Подтверждение удаления',
            f'Вы уверены, что хотите удалить тему "{theme}"?',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            output_file = 'analysis_results.json'

            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    try:
                        existing_data = json.load(f)
                    except json.JSONDecodeError:
                        existing_data = {}

                if theme in existing_data:
                    del existing_data[theme]

                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(existing_data, f, ensure_ascii=False, indent=4)

                    self.load_keywords()
                else:
                    self.result_label.setText(f"Тема '{theme}' не найдена.")
            else:
                self.result_label.setText("Файл с ключевыми фразами не найден.")

    def init_analysis_tab(self):
        layout = QVBoxLayout()

        self.button_load_docx = QPushButton('Загрузить .docx файлы', self)
        self.button_load_docx.clicked.connect(self.load_docx)
        layout.addWidget(self.button_load_docx)

        self.file_list_widget = QFormLayout()
        layout.addLayout(self.file_list_widget)

        self.threshold_label = QLabel('Порог сходства: 0.900', self)
        layout.addWidget(self.threshold_label)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 999)
        self.slider.setValue(900)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.update_threshold_label)
        layout.addWidget(self.slider)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.result_label = QLabel('Результаты:')
        layout.addWidget(self.result_label)

        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)

        self.button = QPushButton('Запустить анализ', self)
        self.button.clicked.connect(self.run_analysis)
        layout.addWidget(self.button)

        self.tab_analysis.setLayout(layout)

    def update_threshold_label(self):
        threshold_value = self.slider.value() / 1000.0
        self.threshold_label.setText(f'Порог сходства: {threshold_value:.3f}')

    def load_docx(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Открыть .docx файлы", "", "Word Files (*.docx)")
        if file_paths:
            for file_path in file_paths:
                h_layout = QHBoxLayout()
                checkbox = QCheckBox()
                h_layout.addWidget(checkbox)
                h_layout.addWidget(QLabel(file_path))
                theme_input = QLineEdit()
                h_layout.addWidget(theme_input)
                self.file_list_widget.addRow(h_layout)

    def run_analysis(self):
        files_data = []
        for index in range(self.file_list_widget.count()):
            h_layout = self.file_list_widget.itemAt(index).layout()
            checkbox = h_layout.itemAt(0).widget()
            theme_input = h_layout.itemAt(2).widget()

            if checkbox.isChecked():
                theme = theme_input.text()
                if theme:
                    file_path = h_layout.itemAt(1).widget().text()
                    files_data.append((file_path, theme))

        if not files_data:
            self.result_label.setText("Пожалуйста, выберите файлы для анализа.")
            return

        self.result_text.clear()
        threshold = self.slider.value() / 1000.0

        self.progress_bar.show()
        self.thread = AnalysisThread(files_data, threshold)
        self.thread.progress.connect(self.update_progress)
        self.thread.new_result.connect(self.append_result)
        self.thread.analysis_finished.connect(self.save_results)
        self.thread.analysis_finished.connect(self.hide_progress_bar)
        self.progress_bar.setValue(0)
        self.thread.start()

    def hide_progress_bar(self):
        self.progress_bar.hide()

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def append_result(self, result):
        current_text = self.result_text.toPlainText()
        updated_text = current_text + result + '\n'
        self.result_text.setPlainText(updated_text)

    def save_results(self, results):
        output_file = 'analysis_results.json'

        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                try:
                    existing_data = json.load(f)
                except json.JSONDecodeError:
                    existing_data = {}
        else:
            existing_data = {}

        total_added_phrases = 0
        for theme, phrases in results.items():
            if theme not in existing_data:
                existing_data[theme] = []

            existing_phrases = {phrase[0] for phrase in existing_data[theme]}
            for phrase in phrases:
                if phrase[0] not in existing_phrases:
                    existing_data[theme].append(phrase)
                    total_added_phrases += 1

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=4)

        self.result_label.setText(f"Сохранено элементов: {total_added_phrases}")
        self.load_keywords()
        self.progress_bar.setValue(100)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
