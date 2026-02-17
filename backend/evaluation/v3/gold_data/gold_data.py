from dataclasses import dataclass
from typing import Literal
from haystack import Document

Difficulty = Literal["easy", "medium", "hard"]


@dataclass
class GoldData:
    question: str
    expected_answer: str
    expected_context: list[str]
    difficulty: Difficulty


GOLD_DATA: list[GoldData] = [
    #
    # DIFFICULTY : EASY
    #
    GoldData(
        question="What happened in the school bus accident when John was a teenager?",
        expected_answer="The school bus skidded on wet pavement and collided with a fallen tree. John's best friend Tommy suffered a broken arm and the bus driver was hospitalized. John emerged unscathed but philosophically unaffected.",
        expected_context=["the-absurd-adolescence-of-john-doe.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What caused the fire that destroyed John's house?",
        expected_answer="The fire was caused by a faulty wire in one of his beloved antique clocks.",
        expected_context=["the-calm-storm-of-john-doe.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What happened to John at the library when he was 21?",
        expected_answer="He lost his job as a junior assistant due to budget cuts ('last in, first out'). Three positions were eliminated including his.",
        expected_context=["the-library-of-small-disappointments.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What happened at John's wedding?",
        expected_answer="His fiancée Lila called off the wedding the night before due to panic. John still went to the church, waited, then bought a pretzel and fed it to pigeons.",
        expected_context=["the-wedding-that-wasnt.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="How old was John when the school bus accident happened?",
        expected_answer="John was fifteen when the school bus accident occurred.",
        expected_context=["the-absurd-adolescence-of-john-doe.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What injury did Tommy suffer in the bus accident?",
        expected_answer="Tommy suffered a broken arm in the bus accident.",
        expected_context=["the-absurd-adolescence-of-john-doe.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What did John call the club he started after the bus accident?",
        expected_answer="John started a 'Bus Philosophy Club' after the bus accident.",
        expected_context=["the-absurd-adolescence-of-john-doe.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What did John use to dry his books after the bus accident?",
        expected_answer="John dried his books with a hairdryer after the bus accident.",
        expected_context=["the-absurd-adolescence-of-john-doe.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What profession did John have when his house burned down?",
        expected_answer="John worked as a librarian when his house burned down.",
        expected_context=["the-calm-storm-of-john-doe.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What did John collect that caused the house fire?",
        expected_answer="John collected antique clocks, and a faulty wire in one caused the house fire.",
        expected_context=["the-calm-storm-of-john-doe.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What did John do with the pigeons after the house fire?",
        expected_answer="John fed crumbs from a stale sandwich to pigeons after the house fire.",
        expected_context=["the-calm-storm-of-john-doe.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What type of clocks did John start collecting after the fire?",
        expected_answer="John started collecting digital clocks after the fire.",
        expected_context=["the-calm-storm-of-john-doe.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What book was John packing last in the archive?",
        expected_answer="John was packing a 17th-century edition of Montaigne’s Essays last.",
        expected_context=["the-last-book.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What phrase was in the last book John packed?",
        expected_answer="The phrase 'Que sçay-je?' (What do I know?) was in the last book.",
        expected_context=["the-last-book.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="How long did John spend packing volumes in the archive?",
        expected_answer="John spent the last three weeks packing volumes in the archive.",
        expected_context=["the-last-book.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What did John keep from the archive as a private act?",
        expected_answer="John kept the call number of Montaigne’s Essays as a private act.",
        expected_context=["the-last-book.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="Who was the head librarian at Evergreen Public Library?",
        expected_answer="Mrs. Hargrove was the head librarian at Evergreen Public Library.",
        expected_context=["the-library-of-small-disappointments.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="How many positions were eliminated at the library?",
        expected_answer="Three positions were eliminated at the library.",
        expected_context=["the-library-of-small-disappointments.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What was the rare-book dealer's unusual requirement?",
        expected_answer="The rare-book dealer required a handwritten essay on why dust is noble.",
        expected_context=["the-library-of-small-disappointments.txt"],
        difficulty="easy",
    ),
    GoldData(
        question="What did John note about dust in his new job?",
        expected_answer="John decided that the dust was rather noble in his new job.",
        expected_context=["the-library-of-small-disappointments.txt"],
        difficulty="easy",
    ),
    #
    # DIFFICULTY : MEDIUM
    #
    GoldData(
        question="How did John react to the school bus accident compared to others?",
        expected_answer="While others were shaken and counselors offered group hugs, John reacted with philosophical detachment. He saw it as 'the universe playing bumper cars with reality' and started a 'Bus Philosophy Club' to discuss existentialism.",
        expected_context=["the-absurd-adolescence-of-john-doe.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="How did John react when his house burned down?",
        expected_answer="He remained calm and philosophical, saying 'it seems the house has decided to retire early' and noting 'Fire sales are literal.' He wasn't distraught and rebuilt his life with 'absurd deliberation.'",
        expected_context=["the-calm-storm-of-john-doe.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="Why did the university archive close and what book was John packing last?",
        expected_answer="The university decided to digitize everything and close the physical stacks. The last book John packed was a 17th-century edition of Montaigne's Essays with the phrase 'Que sçay-je?' (What do I know?).",
        expected_context=["the-last-book.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What settlement notice did John receive when he was 53?",
        expected_answer="A class-action settlement from a chemical plant that leaked into Evergreen's water supply years earlier. It offered modest compensation for potential long-term health effects.",
        expected_context=["the-ordinary-tuesday.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What happened to John's sister Mara?",
        expected_answer="She was diagnosed with an aggressive form of leukemia and given months to live. John traveled to be with her, read to her, and was present when she died three weeks later.",
        expected_context=["the-quiet-clockwork-of-john-doe.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What was John's diagnosis at age 32 and how did he react?",
        expected_answer="He was diagnosed with stage IV pancreatic cancer. He reacted with calm acceptance, asking only 'How long until the coffee stops tasting like coffee?' and continued his routines with minor adjustments.",
        expected_context=["the-uninvited-guest.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What did John do the day after losing his library job?",
        expected_answer="The day after losing his job, John sat at his desk and wrote notes, then planned to apply for three more jobs, one of which he expected to be ridiculous.",
        expected_context=["the-library-of-small-disappointments.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="How did Mara die and what was John's immediate action?",
        expected_answer="Mara died quietly while John was reading to her from a chapter on the indifference of the stars. He finished the paragraph before closing the book.",
        expected_context=["the-quiet-clockwork-of-john-doe.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What did John say about fear to Mara?",
        expected_answer="John said to Mara that fear is just the body’s last honest opinion, but she would find out whether the absurd has an ending or just another act.",
        expected_context=["the-quiet-clockwork-of-john-doe.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What did John do with Mara's hospital bracelet?",
        expected_answer="John added Mara's hospital bracelet to his mementos box labeled 'Act III' after her death.",
        expected_context=["the-quiet-clockwork-of-john-doe.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="How did the oncologist describe John's prognosis?",
        expected_answer="The oncologist described John's prognosis as statistics that hovered between 'slim' and 'theoretical' for stage IV pancreatic cancer.",
        expected_context=["the-uninvited-guest.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What adjustments did John make after his diagnosis?",
        expected_answer="John adjusted his routine by fifteen minutes each morning to accommodate the new fatigue and continued his routines with minor adjustments.",
        expected_context=["the-uninvited-guest.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What note did John leave in his cupboard?",
        expected_answer="John left a note saying: 'If you’re reading this, the coffee is probably stale. Make a new pot anyway. Life is absurd enough without drinking yesterday’s grounds.'",
        expected_context=["the-uninvited-guest.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What did John do after the wedding was called off?",
        expected_answer="John went to the church, waited twenty minutes for a miracle, then walked out, bought a pretzel, ate half, and fed the rest to pigeons.",
        expected_context=["the-wedding-that-wasnt.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What did John say to the vendor about the rose?",
        expected_answer="John said the rose worked perfectly, as it reminded him that beauty doesn’t always come with permanence.",
        expected_context=["the-wedding-that-wasnt.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What did John keep from the archive and why?",
        expected_answer="John kept a slip of paper with the call number of Montaigne’s Essays as a small, private act of defiance against total digitization.",
        expected_context=["the-last-book.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What did John say about missing the books?",
        expected_answer="John said missing is just another way of remembering, and remembering is free.",
        expected_context=["the-last-book.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What did John use the settlement money for?",
        expected_answer="John used the settlement money to buy a better coffee maker.",
        expected_context=["the-ordinary-tuesday.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="What did John think about fighting entropy?",
        expected_answer="John thought fighting entropy is like yelling at gravity—undignified.",
        expected_context=["the-ordinary-tuesday.txt"],
        difficulty="medium",
    ),
    GoldData(
        question="How did John cope with the house fire and what metaphor did he use?",
        expected_answer="John coped by accepting it calmly, using the metaphor that life's a library where books burn but stories float around waiting for new shelves.",
        expected_context=["the-calm-storm-of-john-doe.txt"],
        difficulty="medium",
    ),
    #
    # DIFFICULTY : HARD
    #
    GoldData(
        question="What jobs did John apply for after losing his library job?",
        expected_answer="Night security at a self-storage facility, part-time barista at a shop that only served oat-milk lattes, and cataloguing assistant at a private rare-book dealer who required a handwritten essay on why dust is noble. He took the rare-book job.",
        expected_context=["the-library-of-small-disappointments.txt"],
        difficulty="hard",
    ),
    GoldData(
        question="How did John handle his sister's death?",
        expected_answer="He remained stoic, finished reading the paragraph he was on when she died, sat with her for a while, then added her hospital bracelet to his mementos box labeled 'Act III' and continued with life.",
        expected_context=["the-quiet-clockwork-of-john-doe.txt"],
        difficulty="hard",
    ),
    GoldData(
        question="What were John's last requests?",
        expected_answer="A fresh pot of terrible coffee and that someone finish cataloguing the unfinished boxes of uncatalogued letters in the archive basement.",
        expected_context=["the-uninvited-guest.txt"],
        difficulty="hard",
    ),
    GoldData(
        question="What was John's philosophy about tragedy throughout his life?",
        expected_answer="He saw tragedy as life's way of reminding us that control is an illusion, the universe's way of editing the plot, or simply 'the universe playing bumper cars with reality.' He believed in accepting absurdity rather than fighting it.",
        expected_context=[
            "the-absurd-adolescence-of-john-doe.txt",
            "the-calm-storm-of-john-doe.txt",
            "the-uninvited-guest.txt",
        ],
        difficulty="hard",
    ),
    GoldData(
        question="What jobs did John have throughout his life?",
        expected_answer="Junior assistant at Evergreen Public Library, cataloguing assistant at a private rare-book dealer, rare book cataloguer at university archive, librarian, and eventually owner of a secondhand bookstore.",
        expected_context=[
            "the-library-of-small-disappointments.txt",
            "the-last-book.txt",
            "the-quiet-clockwork-of-john-doe.txt",
            "the-ordinary-tuesday.txt",
        ],
        difficulty="hard",
    ),
    GoldData(
        question="How did John's reactions to the bus accident and house fire reflect his philosophy?",
        expected_answer="In both the bus accident and house fire, John reacted with philosophical detachment and calm acceptance, seeing them as the universe's absurd events rather than personal tragedies, starting a philosophy club after the accident and rebuilding deliberately after the fire.",
        expected_context=[
            "the-absurd-adolescence-of-john-doe.txt",
            "the-calm-storm-of-john-doe.txt",
        ],
        difficulty="hard",
    ),
    GoldData(
        question="What common theme connects John's job loss at the library and the archive closure?",
        expected_answer="Both events involved budget cuts or modernization leading to job changes; John lost his junior assistant role to cuts and the archive closed for digitization, but he adapted by finding new roles in rare books and eventually owning a bookstore.",
        expected_context=[
            "the-library-of-small-disappointments.txt",
            "the-last-book.txt",
        ],
        difficulty="hard",
    ),
    GoldData(
        question="How did John's handling of his sister's death compare to his own cancer diagnosis?",
        expected_answer="John handled both with stoicism and calm; he finished reading to Mara and labeled her bracelet 'Act III,' and for his cancer, he adjusted routines minimally and saw it as a predictable plot.",
        expected_context=[
            "the-quiet-clockwork-of-john-doe.txt",
            "the-uninvited-guest.txt",
        ],
        difficulty="hard",
    ),
    GoldData(
        question="What role did coffee play in John's life across events?",
        expected_answer="Coffee was a constant ritual; he made terrible coffee during his cancer, requested terrible coffee as a last wish, and used settlement money for a better coffee maker, symbolizing normalcy amid tragedy.",
        expected_context=["the-uninvited-guest.txt", "the-ordinary-tuesday.txt"],
        difficulty="hard",
    ),
    GoldData(
        question="How did John's wedding cancellation and job losses show his resilience?",
        expected_answer="John treated the wedding cancellation like a plot twist, feeding pigeons afterward, and after job losses from budget cuts and archive closure, he applied for new roles and adapted without distress.",
        expected_context=[
            "the-wedding-that-wasnt.txt",
            "the-library-of-small-disappointments.txt",
            "the-last-book.txt",
        ],
        difficulty="hard",
    ),
    GoldData(
        question="What environmental event linked to John's settlement, and how did it tie to his health?",
        expected_answer="The chemical plant leak into Evergreen's water supply led to the settlement for potential health effects, which subtly connected to his later cancer diagnosis, though he accepted both without fury.",
        expected_context=["the-ordinary-tuesday.txt", "the-uninvited-guest.txt"],
        difficulty="hard",
    ),
    GoldData(
        question="How did John's philosophy evolve from adolescence to death?",
        expected_answer="From seeing accidents as 'bumper cars with reality' in adolescence, to accepting fire as 'editing the plot,' to viewing his cancer as a 'predictable plot' with curiosity about the ending, John consistently embraced life's absurdity.",
        expected_context=[
            "the-absurd-adolescence-of-john-doe.txt",
            "the-calm-storm-of-john-doe.txt",
            "the-uninvited-guest.txt",
        ],
        difficulty="hard",
    ),
    GoldData(
        question="What mementos did John keep from family tragedies?",
        expected_answer="From his sister's death, John kept her hospital bracelet in his 'Act III' mementos box; tragedies like parents' accident and sister's death shaped his stoic philosophy.",
        expected_context=["the-quiet-clockwork-of-john-doe.txt"],
        difficulty="hard",
    ),
    GoldData(
        question="How did books and libraries thread through John's career setbacks?",
        expected_answer="John's library job loss led to rare-book cataloguing, then university archive work ending in digitization, culminating in owning a bookstore; books were constant amid changes.",
        expected_context=[
            "the-library-of-small-disappointments.txt",
            "the-last-book.txt",
            "the-ordinary-tuesday.txt",
        ],
        difficulty="hard",
    ),
    GoldData(
        question="What ironies did John note in his life events?",
        expected_answer="Ironies like a clock causing a fire, digitization closing an archive of physical books, and a water pollution settlement possibly linking to his cancer, all accepted as life's absurdities.",
        expected_context=[
            "the-calm-storm-of-john-doe.txt",
            "the-last-book.txt",
            "the-ordinary-tuesday.txt",
            "the-uninvited-guest.txt",
        ],
        difficulty="hard",
    ),
    GoldData(
        question="How did John's relationships with friends and family show his detachment?",
        expected_answer="John offered observations instead of emotions to friends during cancer, was stoic with Mara during her death, and calmly accepted Lila's wedding cancellation.",
        expected_context=[
            "the-uninvited-guest.txt",
            "the-quiet-clockwork-of-john-doe.txt",
            "the-wedding-that-wasnt.txt",
        ],
        difficulty="hard",
    ),
    GoldData(
        question="What common items symbolized normalcy in John's tragedies?",
        expected_answer="Coffee symbolized normalcy: terrible coffee during cancer, a new maker from settlement, and pretzels/pigeons after wedding, all grounding him in routine.",
        expected_context=[
            "the-uninvited-guest.txt",
            "the-ordinary-tuesday.txt",
            "the-wedding-that-wasnt.txt",
        ],
        difficulty="hard",
    ),
    GoldData(
        question="How did Evergreen feature in John's life story?",
        expected_answer="Evergreen was the suburb of his adolescence, home to his library job, water pollution settlement, and where he owned his bookstore amid various tragedies.",
        expected_context=[
            "the-absurd-adolescence-of-john-doe.txt",
            "the-library-of-small-disappointments.txt",
            "the-ordinary-tuesday.txt",
        ],
        difficulty="hard",
    ),
    GoldData(
        question="What role did reading play in John's coping mechanisms?",
        expected_answer="John read philosophy to Mara, continued reading during chemo, and packed Montaigne's Essays last, using books to process absurdity and mortality.",
        expected_context=[
            "the-quiet-clockwork-of-john-doe.txt",
            "the-uninvited-guest.txt",
            "the-last-book.txt",
        ],
        difficulty="hard",
    ),
    GoldData(
        question="How did John's possessions reflect his philosophy?",
        expected_answer="Clocks (causing fire), mementos box ('Act III'), philosophy books, and coffee rituals all embodied acceptance of time, memory, and life's indifference.",
        expected_context=[
            "the-calm-storm-of-john-doe.txt",
            "the-quiet-clockwork-of-john-doe.txt",
            "the-uninvited-guest.txt",
        ],
        difficulty="hard",
    ),
]


def create_ground_truth_documents_for_question(
    question_data: GoldData,
) -> list[Document]:
    """
    Create ground truth documents for a specific question.
    Returns a list of Haystack Document objects marked as relevant.
    """
    from haystack import Document

    documents = []
    for filename in question_data.expected_context:
        doc = Document(
            content=f"Content from {filename}",
            meta={
                "filename": filename,
                "relevant": True,
                "question": question_data.question,
                "expected_answer": question_data.expected_answer,
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
        ground_truths[item.question] = create_ground_truth_documents_for_question(item)
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
    question_data: GoldData,
) -> list[Document]:
    """
    Create ground truth documents with actual content for more accurate evaluation.
    """
    documents = []
    for filename in question_data.expected_context:
        # Use actual content if available, otherwise use placeholder
        content = DOCUMENT_CONTENTS.get(filename, f"Content from {filename}")

        doc = Document(
            content=content,
            meta={
                "filename": filename,
                "relevant": True,
                "question": question_data.question,
                "expected_answer": question_data.expected_answer,
                "difficulty": question_data.difficulty,
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
        enhanced_ground_truths[item.question] = (
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


easy_count = sum(1 for d in GOLD_DATA if d.difficulty == "easy")
medium_count = sum(1 for d in GOLD_DATA if d.difficulty == "medium")
hard_count = sum(1 for d in GOLD_DATA if d.difficulty == "hard")
print(
    f"Total: {len(GOLD_DATA)} | Easy: {easy_count} | Medium: {medium_count} | Hard: {hard_count}"
)
