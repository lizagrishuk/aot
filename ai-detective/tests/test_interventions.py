import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from interventions import mask_text, remove_fact, MARKER, claim_states_fact  # noqa: E402


def test_mask_text_masks_whole_sentences_and_log_lines():
    text = "Первое предложение. Второе про 18:40 здесь. Третье.\n2026-03-14T18:40:03 — delay\n2026-03-14T18:41:08 — other"
    out, n = mask_text(text, ["18:40"])
    assert n == 2 and "18:40" not in out and "Первое предложение." in out and "other" in out and MARKER in out


def test_registry_in_mode_a_drops_claims_whose_evidence_states_the_fact():
    claim = {"claim_id": "C26", "text": "Счётчик равнялся 24 событиям при одном уникальном сообщении.",
             "evidence": "2026-03-14T18:42:15 — moderation_counter event_count=24 unique_message_count=1"}
    assert claim_states_fact(claim, ["unique_message_count"])
    ctx = "[D07] Журнал\n2026-03-14T18:42:15 — moderation_counter event_count=24 unique_message_count=1\n\nКОМПАКТНЫЙ РЕЕСТР ID УТВЕРЖДЕНИЙ (источники выше)\n[C26; D07; direct] Счётчик равнялся 24 событиям при одном уникальном сообщении.\n[C01; D01; direct] Другое."
    out, dropped, masked = remove_fact(ctx, ["unique_message_count"], "A", [claim])
    assert dropped == ["C26"] and "unique_message_count" not in out and "[C01;" in out and masked == 1


def test_claim_with_partly_matching_evidence_is_masked_not_dropped_in_flat_context():
    line = '[C22; source=D06; status=reported] Илья связал события с сообщением. Evidence: "Связал. Задержка началась до блокировки. Причина похожа на сбой."'
    out, dropped, masked = remove_fact(line, ["началась до блокировки"], "B")
    assert dropped == [] and masked == 1 and "началась до блокировки" not in out and "[C22;" in out


def test_empty_branch_header_is_removed_in_mode_c():
    ctx = 'ВЕТВЬ [0, 0, 0] (x=0):\n[C24; source=D07; status=direct] Задержка 18:40. Evidence: "18:40:03 — delay"\nВЕТВЬ [1, 1, 1] (x=13):\n[C11; source=D04; status=reported] Другое. Evidence: "Другое."'
    out, dropped, _ = remove_fact(ctx, ["18:40"], "C")
    assert dropped == ["C24"] and "[0, 0, 0]" not in out and "[1, 1, 1]" in out
