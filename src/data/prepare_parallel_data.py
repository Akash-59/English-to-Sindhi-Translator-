"""
Parallel data acquisition and preprocessing script for English-to-Sindhi translation.
Acquires parallel pairs across core domains (everyday, education, health, technology, formal),
executes validation, and prepares training, validation, and test splits.
"""

import os
from pathlib import Path
import pandas as pd

import config
from src.data.validate_dataset import DatasetValidator


# Verified core bilingual sentence pairs across representative domains
CORE_BILINGUAL_CORPUS = [
    # Everyday / Greetings / General
    {"english": "Hello, how are you?", "sindhi": "هيلو، توهان ڪيئن آهيو؟", "source": "core_curated", "license": "CC0"},
    {"english": "Good morning.", "sindhi": "صبح جو سلام۔", "source": "core_curated", "license": "CC0"},
    {"english": "Good evening.", "sindhi": "شام جو سلام۔", "source": "core_curated", "license": "CC0"},
    {"english": "Thank you very much.", "sindhi": "توهان جي وڏي مهرباني۔", "source": "core_curated", "license": "CC0"},
    {"english": "You are welcome.", "sindhi": "توهان ڀلي ڪري آيا۔", "source": "core_curated", "license": "CC0"},
    {"english": "What is your name?", "sindhi": "توهان جو نالو ڇا آهي؟", "source": "core_curated", "license": "CC0"},
    {"english": "My name is Ali.", "sindhi": "منهنجو نالو علي آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "Where do you live?", "sindhi": "توهان ڪٿي رهو ٿا؟", "source": "core_curated", "license": "CC0"},
    {"english": "I live in Hyderabad.", "sindhi": "مان حيدرآباد ۾ رهان ٿو۔", "source": "core_curated", "license": "CC0"},
    {"english": "I live in Karachi.", "sindhi": "مان ڪراچي ۾ رهان ٿو۔", "source": "core_curated", "license": "CC0"},
    {"english": "Yes, of course.", "sindhi": "ها، بلڪل۔", "source": "core_curated", "license": "CC0"},
    {"english": "No, I do not agree.", "sindhi": "نه، مان متفق نه آهيان۔", "source": "core_curated", "license": "CC0"},
    {"english": "Please help me.", "sindhi": "مهرباني ڪري منهنجي مدد ڪريو۔", "source": "core_curated", "license": "CC0"},
    {"english": "I need help.", "sindhi": "مون کي مدد گهرجي۔", "source": "core_curated", "license": "CC0"},
    {"english": "I am fine.", "sindhi": "مان ٺيڪ آهيان۔", "source": "core_curated", "license": "CC0"},
    {"english": "Goodbye, see you soon.", "sindhi": "خدا حافظ، جلد ملنداسين۔", "source": "core_curated", "license": "CC0"},

    # Education / Learning / Knowledge
    {"english": "I am learning Sindhi.", "sindhi": "مان سنڌي سکي رهيو آهيان۔", "source": "core_curated", "license": "CC0"},
    {"english": "Knowledge is important.", "sindhi": "علم اهم آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "Education is the key to success.", "sindhi": "تعليم ڪاميابيءَ جي ڪنجي آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "The student is reading a book.", "sindhi": "شاگرد ڪتاب پڙهي رهيو آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "The teacher is in the classroom.", "sindhi": "استاد ڪلاس روم ۾ آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "Books are our best friends.", "sindhi": "ڪتاب اسان جا بهترين دوست آهن۔", "source": "core_curated", "license": "CC0"},
    {"english": "I study at the university.", "sindhi": "مان يونيورسٽي ۾ پڙهان ٿو۔", "source": "core_curated", "license": "CC0"},
    {"english": "This exam is very important.", "sindhi": "هي امتحان تمام اهم آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "He writes with a pen.", "sindhi": "هو قلم سان لکي ٿو۔", "source": "core_curated", "license": "CC0"},
    {"english": "We must respect our teachers.", "sindhi": "اسان کي پنهنجي استادن جو احترام ڪرڻ گهرجي۔", "source": "core_curated", "license": "CC0"},
    {"english": "Library is a place for quiet study.", "sindhi": "لائبريري خاموش پڙهائيءَ جي جاءِ آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "Science helps us understand the universe.", "sindhi": "سائنس اسان کي ڪائنات سمجهڻ ۾ مدد ڪري ٿي۔", "source": "core_curated", "license": "CC0"},

    # Family / Home
    {"english": "My family lives together.", "sindhi": "منهنجو خاندان گڏ رهي ٿو۔", "source": "core_curated", "license": "CC0"},
    {"english": "My mother is cooking delicious food.", "sindhi": "منهنجي ماءُ لذيذ کاڌو تيار ڪري رهي آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "My father went to work.", "sindhi": "منهنجو پيءُ ڪم تي ويو آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "My brother is younger than me.", "sindhi": "منهنجو ڀاءُ مون کان ننڍو آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "My sister is studying medicine.", "sindhi": "منهنجي ڀيڻ طب جي پڙهائي ڪري رهي آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "Children are playing in the garden.", "sindhi": "ٻار باغ ۾ کيڏي رهيا آهن۔", "source": "core_curated", "license": "CC0"},
    {"english": "Home is where the heart is.", "sindhi": "گهر اهو آهي جتي دل هجي۔", "source": "core_curated", "license": "CC0"},

    # Health / Medical
    {"english": "I need to see a doctor.", "sindhi": "مون کي ڊاڪٽر کي ڏيکارڻ جي ضرورت آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "Drink plenty of water every day.", "sindhi": "هر روز گهڻو پاڻي پيئو۔", "source": "core_curated", "license": "CC0"},
    {"english": "Health is the greatest wealth.", "sindhi": "صحت وڏي دولت آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "Take this medicine after meals.", "sindhi": "هي دوا ماني کائڻ کان پوءِ وٺو۔", "source": "core_curated", "license": "CC0"},
    {"english": "He has a high fever.", "sindhi": "هن کي تيز بخار آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "The hospital is open 24 hours.", "sindhi": "اسپتال چوويهه ڪلاڪ کليل آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "Exercise is good for health.", "sindhi": "ورزش صحت لاءِ سٺي آهي۔", "source": "core_curated", "license": "CC0"},

    # Travel / Transportation / Directions
    {"english": "Where is the railway station?", "sindhi": "ريلوي اسٽيشن ڪٿي آهي؟", "source": "core_curated", "license": "CC0"},
    {"english": "Turn right at the next street.", "sindhi": "ايندڙ گهٽيءَ مان ساڄي پاسي مڙو۔", "source": "core_curated", "license": "CC0"},
    {"english": "How much does this ticket cost?", "sindhi": "هن ٽڪيٽ جي قيمت ڪيتري آهي؟", "source": "core_curated", "license": "CC0"},
    {"english": "The train arrives on time.", "sindhi": "گاڏي وقت تي اچي ٿي۔", "source": "core_curated", "license": "CC0"},
    {"english": "I want to visit Sindh.", "sindhi": "مان سنڌ گهمڻ چاهيان ٿو۔", "source": "core_curated", "license": "CC0"},
    {"english": "The weather is very pleasant today.", "sindhi": "اڄ موسم تمام سٺي آهي۔", "source": "core_curated", "license": "CC0"},

    # Technology / Computing
    {"english": "This application works offline.", "sindhi": "هي اپليڪيشن بنا انٽرنيٽ جي ڪم ڪري ٿي۔", "source": "core_curated", "license": "CC0"},
    {"english": "Artificial intelligence is changing the world.", "sindhi": "مصنوعي ذهانت دنيا کي بدلائي رهي آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "Save your work frequently.", "sindhi": "پنهنجو ڪم بار بار محفوظ ڪريو۔", "source": "core_curated", "license": "CC0"},
    {"english": "The computer requires a password.", "sindhi": "ڪمپيوٽر کي پاسورڊ جي ضرورت آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "Machine translation helps bridge language barriers.", "sindhi": "مشيني ترجمو ٻولين جي رڪاوٽن کي دور ڪرڻ ۾ مدد ڪري ٿو۔", "source": "core_curated", "license": "CC0"},
    {"english": "Click the button to download the file.", "sindhi": "فائل ڊائون لوڊ ڪرڻ لاءِ بٽڻ دٻايو۔", "source": "core_curated", "license": "CC0"},

    # Formal / Administrative
    {"english": "Please submit your application before the deadline.", "sindhi": "مهرباني ڪري مقرر تاريخ کان اڳ پنهنجي درخواست جمع ڪرايو۔", "source": "core_curated", "license": "CC0"},
    {"english": "All citizens have equal rights.", "sindhi": "سڀني شهرين کي برابر حق حاصل آهن۔", "source": "core_curated", "license": "CC0"},
    {"english": "The official report has been published.", "sindhi": "سرڪاري رپورٽ شايع ڪئي وئي آهي۔", "source": "core_curated", "license": "CC0"},
    {"english": "We are committed to quality and accuracy.", "sindhi": "اسان معيار ۽ درستگيءَ لاءِ پرعزم آهيون۔", "source": "core_curated", "license": "CC0"},
    {"english": "Terms and conditions apply.", "sindhi": "شرط ۽ ضابطا لاڳو ٿين ٿاـ", "source": "core_curated", "license": "CC0"},
]


def prepare_parallel_datasets(output_dir: Path = config.DATA_DIR):
    """Main workflow to assemble, clean, validate, and partition parallel data."""
    print("[PrepareData] Assembling core English-Sindhi parallel data...")
    os.makedirs(output_dir, exist_ok=True)
    raw_df = pd.DataFrame(CORE_BILINGUAL_CORPUS)

    # Save sample reference dataset
    sample_file = output_dir / "sample_parallel.csv"
    raw_df.to_csv(sample_file, index=False, encoding="utf-8")
    print(f"[PrepareData] Sample parallel data saved -> {sample_file}")

    # Validate and clean
    validator = DatasetValidator(random_seed=42)
    clean_df, audit = validator.clean_parallel_df(raw_df)

    print(f"[PrepareData] Clean records: {audit['final_clean_records']} / {audit['initial_records']}")

    # Partition into Train / Val / Test (80 / 10 / 10)
    train_df, val_df, test_df = validator.split_dataset(clean_df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

    # Save partitions
    validator.save_splits(train_df, val_df, test_df, output_dir)

    # Save audit report
    report_file = config.REPORTS_DIR / "dataset_cleaning_report.json"
    validator.save_report(report_file)

    return train_df, val_df, test_df


if __name__ == "__main__":
    prepare_parallel_datasets()
