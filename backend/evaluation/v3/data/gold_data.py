from haystack import Document

GOLD_DATA = [
    {
        "question": "What happened in the school bus accident when John was a teenager?",
        "expected_answer": "The school bus skidded on wet pavement and collided with a fallen tree. John's best friend Tommy suffered a broken arm and the bus driver was hospitalized. John emerged unscathed but philosophically unaffected.",
        "expected_context": ["the-absurd-adolescence-of-john-doe.txt"],
        "difficulty": "easy",
    },
    {
        "question": "How did John react to the school bus accident compared to others?",
        "expected_answer": "While others were shaken and counselors offered group hugs, John reacted with philosophical detachment. He saw it as 'the universe playing bumper cars with reality' and started a 'Bus Philosophy Club' to discuss existentialism.",
        "expected_context": ["the-absurd-adolescence-of-john-doe.txt"],
        "difficulty": "medium",
    },
    {
        "question": "What caused the fire that destroyed John's house?",
        "expected_answer": "The fire was caused by a faulty wire in one of his beloved antique clocks.",
        "expected_context": ["the-calm-storm-of-john-doe.txt"],
        "difficulty": "easy",
    },
    {
        "question": "How did John react when his house burned down?",
        "expected_answer": "He remained calm and philosophical, saying 'it seems the house has decided to retire early' and noting 'Fire sales are literal.' He wasn't distraught and rebuilt his life with 'absurd deliberation.'",
        "expected_context": ["the-calm-storm-of-john-doe.txt"],
        "difficulty": "medium",
    },
    {
        "question": "Why did the university archive close and what book was John packing last?",
        "expected_answer": "The university decided to digitize everything and close the physical stacks. The last book John packed was a 17th-century edition of Montaigne's Essays with the phrase 'Que sçay-je?' (What do I know?).",
        "expected_context": ["the-last-book.txt"],
        "difficulty": "medium",
    },
    {
        "question": "What happened to John at the library when he was 21?",
        "expected_answer": "He lost his job as a junior assistant due to budget cuts ('last in, first out'). Three positions were eliminated including his.",
        "expected_context": ["the-library-of-small-disappointments.txt"],
        "difficulty": "easy",
    },
    {
        "question": "What jobs did John apply for after losing his library job?",
        "expected_answer": "Night security at a self-storage facility, part-time barista at a shop that only served oat-milk lattes, and cataloguing assistant at a private rare-book dealer who required a handwritten essay on why dust is noble. He took the rare-book job.",
        "expected_context": ["the-library-of-small-disappointments.txt"],
        "difficulty": "hard",
    },
    {
        "question": "What settlement notice did John receive when he was 53?",
        "expected_answer": "A class-action settlement from a chemical plant that leaked into Evergreen's water supply years earlier. It offered modest compensation for potential long-term health effects.",
        "expected_context": ["the-ordinary-tuesday.txt"],
        "difficulty": "medium",
    },
    {
        "question": "What happened to John's sister Mara?",
        "expected_answer": "She was diagnosed with an aggressive form of leukemia and given months to live. John traveled to be with her, read to her, and was present when she died three weeks later.",
        "expected_context": ["the-quiet-clockwork-of-john-doe.txt"],
        "difficulty": "medium",
    },
    {
        "question": "How did John handle his sister's death?",
        "expected_answer": "He remained stoic, finished reading the paragraph he was on when she died, sat with her for a while, then added her hospital bracelet to his mementos box labeled 'Act III' and continued with life.",
        "expected_context": ["the-quiet-clockwork-of-john-doe.txt"],
        "difficulty": "hard",
    },
    {
        "question": "What was John's diagnosis at age 32 and how did he react?",
        "expected_answer": "He was diagnosed with stage IV pancreatic cancer. He reacted with calm acceptance, asking only 'How long until the coffee stops tasting like coffee?' and continued his routines with minor adjustments.",
        "expected_context": ["the-uninvited-guest.txt"],
        "difficulty": "medium",
    },
    {
        "question": "What were John's last requests?",
        "expected_answer": "A fresh pot of terrible coffee and that someone finish cataloguing the unfinished boxes of uncatalogued letters in the archive basement.",
        "expected_context": ["the-uninvited-guest.txt"],
        "difficulty": "hard",
    },
    {
        "question": "What happened at John's wedding?",
        "expected_answer": "His fiancée Lila called off the wedding the night before due to panic. John still went to the church, waited, then bought a pretzel and fed it to pigeons.",
        "expected_context": ["the-wedding-that-wasnt.txt"],
        "difficulty": "easy",
    },
    {
        "question": "What was John's philosophy about tragedy throughout his life?",
        "expected_answer": "He saw tragedy as life's way of reminding us that control is an illusion, the universe's way of editing the plot, or simply 'the universe playing bumper cars with reality.' He believed in accepting absurdity rather than fighting it.",
        "expected_context": [
            "the-absurd-adolescence-of-john-doe.txt",
            "the-calm-storm-of-john-doe.txt",
            "the-uninvited-guest.txt",
        ],
        "difficulty": "hard",
    },
    {
        "question": "What jobs did John have throughout his life?",
        "expected_answer": "Junior assistant at Evergreen Public Library, cataloguing assistant at a private rare-book dealer, rare book cataloguer at university archive, librarian, and eventually owner of a secondhand bookstore.",
        "expected_context": [
            "the-library-of-small-disappointments.txt",
            "the-last-book.txt",
            "the-quiet-clockwork-of-john-doe.txt",
            "the-ordinary-tuesday.txt",
        ],
        "difficulty": "hard",
    },
]


def create_ground_truth_documents_for_question(question_data: dict) -> list[Document]:
    """
    Create ground truth documents for a specific question.
    Returns a list of Haystack Document objects marked as relevant.
    """
    from haystack import Document

    documents = []
    for filename in question_data["expected_context"]:
        doc = Document(
            content=f"Content from {filename}",
            meta={
                "filename": filename,
                "relevant": True,
                "question": question_data["question"],
                "expected_answer": question_data["expected_answer"],
            },
        )
        documents.append(doc)
    return documents


def get_all_ground_truth_documents() -> dict:
    """
    Returns a dictionary with question as key and list of ground truth documents as value.
    This is useful for batch evaluation.
    """
    ground_truths = {}
    for item in GOLD_DATA:
        ground_truths[item["question"]] = create_ground_truth_documents_for_question(
            item
        )
    return ground_truths


DOCUMENT_METADATA = {
    "the-absurd-adolescence-of-john-doe.txt": {
        "age": "15",
        "event": "school bus accident",
        "theme": "philosophical detachment",
    },
    "the-calm-storm-of-john-doe.txt": {
        "age": "adult",
        "event": "house fire",
        "theme": "acceptance of loss",
    },
    "the-last-book.txt": {
        "age": "39",
        "event": "archive closure",
        "theme": "digital vs physical",
    },
    "the-library-of-small-disappointments.txt": {
        "age": "21",
        "event": "job loss",
        "theme": "professional setbacks",
    },
    "the-ordinary-tuesday.txt": {
        "age": "53",
        "event": "settlement notice",
        "theme": "life's small ironies",
    },
    "the-quiet-clockwork-of-john-doe.txt": {
        "age": "25",
        "event": "sister's death",
        "theme": "mortality",
    },
    "the-uninvited-guest.txt": {
        "age": "32",
        "event": "cancer diagnosis",
        "theme": "facing death",
    },
    "the-wedding-that-wasnt.txt": {
        "age": "28",
        "event": "called-off wedding",
        "theme": "romantic disappointment",
    },
}

DOCUMENT_CONTENTS = {
    "the-absurd-adolescence-of-john-doe.txt": """
        When John was 15, his school bus skidded on wet pavement and collided with a fallen tree.
        His best friend Tommy suffered a broken arm. The bus driver was hospitalized.
        John emerged unscathed but philosophically unaffected. While others were shaken,
        John saw it as "the universe playing bumper cars with reality" and started a
        "Bus Philosophy Club" to discuss existentialism.
    """,
    "the-calm-storm-of-john-doe.txt": """
        A faulty wire in one of his beloved antique clocks caused a fire that destroyed
        John's house. He remained calm and philosophical, saying "it seems the house
        has decided to retire early" and noting "Fire sales are literal." He wasn't
        distraught and rebuilt his life with "absurd deliberation."
    """,
    "the-last-book.txt": """
        When John was 39, the university decided to digitize everything and close
        the physical stacks where he worked as a rare book cataloguer. The last book
        he packed was a 17th-century edition of Montaigne's Essays with the phrase
        "Que sçay-je?" (What do I know?).
    """,
    "the-library-of-small-disappointments.txt": """
        At 21, John lost his job as a junior assistant at Evergreen Public Library
        due to budget cuts ("last in, first out"). Three positions were eliminated
        including his. He later applied for night security at a self-storage facility,
        part-time barista at a shop that only served oat-milk lattes, and cataloguing
        assistant at a private rare-book dealer who required a handwritten essay on
        why dust is noble. He took the rare-book job.
    """,
    "the-ordinary-tuesday.txt": """
        At 53, John received a class-action settlement from a chemical plant that
        leaked into Evergreen's water supply years earlier. It offered modest
        compensation for potential long-term health effects. John used the money
        to buy a better coffee maker.
    """,
    "the-quiet-clockwork-of-john-doe.txt": """
        When John was 25, his sister Mara was diagnosed with an aggressive form
        of leukemia and given months to live. John traveled to be with her,
        read to her, and was present when she died three weeks later. He remained
        stoic, finished reading the paragraph he was on when she died, sat with
        her for a while, then added her hospital bracelet to his mementos box
        labeled "Act III" and continued with life.
    """,
    "the-uninvited-guest.txt": """
        At 32, John was diagnosed with stage IV pancreatic cancer. He reacted with
        calm acceptance, asking only "How long until the coffee stops tasting like
        coffee?" and continued his routines with minor adjustments. His last requests
        were a fresh pot of terrible coffee and that someone finish cataloguing the
        unfinished boxes of uncatalogued letters in the archive basement.
    """,
    "the-wedding-that-wasnt.txt": """
        At 28, John's fiancée Lila called off the wedding the night before due to
        panic. John still went to the church, waited, then bought a pretzel and
        fed it to pigeons. He saw it as just another plot twist in life's absurd play.
    """,
}


def create_enhanced_ground_truth_documents_for_question(
    question_data: dict,
) -> list[Document]:
    """
    Create ground truth documents with actual content for more accurate evaluation.
    """
    from haystack import Document

    documents = []
    for filename in question_data["expected_context"]:
        # Use actual content if available, otherwise use placeholder
        content = DOCUMENT_CONTENTS.get(filename, f"Content from {filename}")

        doc = Document(
            content=content,
            meta={
                "filename": filename,
                "relevant": True,
                "question": question_data["question"],
                "expected_answer": question_data["expected_answer"],
                "difficulty": question_data.get("difficulty", "medium"),
            },
        )
        documents.append(doc)
    return documents


def get_all_enhanced_ground_truth_documents() -> dict:
    """
    Returns a dictionary with question as key and list of enhanced ground truth documents as value.
    """
    enhanced_ground_truths = {}
    for item in GOLD_DATA:
        enhanced_ground_truths[item["question"]] = (
            create_enhanced_ground_truth_documents_for_question(item)
        )
    return enhanced_ground_truths


def get_ground_truth_for_evaluator(use_enhanced: bool = True) -> dict:
    """
    Get ground truth documents for the evaluator.
    If use_enhanced is True, returns documents with actual content.
    """
    if use_enhanced:
        return get_all_enhanced_ground_truth_documents()
    else:
        return get_all_ground_truth_documents()
