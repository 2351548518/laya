"""the three checkpoints, side by side.

Loads `english`, `multilingual` and `typed-decisions` in turn and asks the same small
question set of an English ticket and a Hindi one, so the choice of checkpoint is visible
before routing is introduced.
"""
import torch

from _common import STATE_EN, STATE_HI, banner, device_line, heading, load

banner("Which checkpoint?", """
    Laya ships three checkpoints with different encoders, windows and training. The names
    do not tell you when to use which, so this example runs the same three questions over
    two tickets: one in English, one in Devanagari Hindi.

      * `english`          421M, ModernBERT-large, 512 tokens -- trained on English.
      * `multilingual`     322M, mmBERT-base, 1024 tokens -- one model for 100+ languages.
      * `typed-decisions`  421M, ModernBERT-large, 1024 tokens -- the English checkpoint
                           fine-tuned on four specific workflows, not a general upgrade.

    Only one checkpoint is resident at a time: each is loaded, used for both tickets, then
    released, because all three together are over a billion parameters.
    """)

TICKETS = [("English ticket", STATE_EN), ("Hindi ticket", STATE_HI)]

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "Which department should handle this request?",
        "criteria": {"billing": "invoices, payments, refunds",
                     "technical": "bugs, outages, system errors",
                     "sales": "pricing, new contracts",
                     "other": "everything else"},
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "criteria": ["not urgent", "soon", "critical deadline or blocking issue"],
    },
    "refund_requested": {
        "type": "noul",
        "instructions": "Does the user explicitly request a refund?",
    },
}

NAMES = ("english", "multilingual", "typed-decisions")
results = {}
specs = {}

for name in NAMES:
    agent = load(name)
    specs[name] = (agent.cfg["encoder"], agent.cfg["max_len"],
                   sum(p.numel() for p in agent.model.parameters()))
    for label, state in TICKETS:
        results[(label, name)] = agent.predict(state, QUESTIONS)["answers"]
    if name == NAMES[0]:
        device_line(agent)
    device = agent.device.type
    del agent
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()

heading("checkpoints")
for name in NAMES:
    encoder, window, params = specs[name]
    print("   %-16s %-28s %5d tokens   %.0fM params" % (name, encoder, window, params / 1e6))


def summary(answers, qid):
    """One compact field per question: the chosen label and its confidence."""
    answer = answers[qid]
    if answer["type"] == "choice":
        return "%-12s p=%.3f conf=%.3f" % (answer["choice"],
                                           answer["probabilities"][answer["choice"]],
                                           answer["confidence"])
    if answer["type"] == "score":
        return "%.2f/%d       conf=%.3f" % (answer["score"], len(answer["probabilities"]) - 1,
                                            answer["confidence"])
    return "%-5s P=%.3f conf=%.3f" % ("true" if answer["noul"] > 0.5 else "false",
                                      answer["noul"], answer["confidence"])


heading("the same questions, per checkpoint")
for label, _ in TICKETS:
    print("   %s" % label)
    for name in NAMES:
        answers = results[(label, name)]
        print("     %-16s department  %s" % (name, summary(answers, "department")))
        print("     %-16s urgency     %s" % ("", summary(answers, "urgency")))
        print("     %-16s refund_req  %s" % ("", summary(answers, "refund_requested")))

print("""
   The table shows the actual decisions for these two tickets. Use held-out tickets
   to measure accuracy across languages. `route_without_loading.py` and `multilingual_routing.py` show how to select a
   checkpoint from the input language, and `typed_decision_workflow.py` runs a typed workflow.
   """)
