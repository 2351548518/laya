"""compare individual calls with `predict_batch()` on classic text.

Apply the same question set to twenty tickets and compare the request time.
"""
import time

from _common import banner, device_line, heading, load

banner("Twenty tickets with predict_batch()", """
    `predict(state, questions)` takes one state. `predict_batch(states, questions)`
    uses the same questions for each input and returns results in input order.
    This example compares a loop with batches of up to four classic text states.
    """)

agent = load("english")
device_line(agent)

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
        "instructions": "Does the customer explicitly ask for a refund?",
    },
}

BODIES = [
    "I was charged twice this month and want the duplicate refunded today.",
    "The API returns 502 errors for our whole team; production is down.",
    "Can you send enterprise pricing and a security questionnaire?",
    "How do I download the PDF invoice for February?",
    "My password reset email never arrives, so I cannot sign in.",
    "Please cancel my subscription at the end of the month.",
    "The export button only offers PDF; CSV would help a lot.",
    "We need two more seats on our plan and a new quote.",
    "The tracking page has not updated in four days.",
    "A refund for order 5521 has not appeared on my statement.",
    "The dashboard is slow since the last release.",
    "Do you offer a discount for annual billing?",
    "I received someone else's invoice in my inbox.",
    "Two-factor authentication codes are rejected on my phone.",
    "Please upgrade us to the business tier this week.",
    "The mobile app crashes whenever I open the reports tab.",
    "Our card was declined even though the bank approved it.",
    "Where can I change the billing contact on the account?",
    "The nightly sync job has failed for three nights running.",
    "I want a full refund and I am disputing the charge with my bank.",
]
TICKETS = [{"ticket": "T-%02d" % (i + 1), "body": body} for i, body in enumerate(BODIES)]

agent.predict(TICKETS[0], QUESTIONS)
agent.predict_batch(TICKETS[:4], QUESTIONS, batch_size=4)

heading("one predict() call per ticket")
t0 = time.perf_counter()
single = [agent.predict(state, QUESTIONS) for state in TICKETS]
loop_s = time.perf_counter() - t0
n_answers = len(TICKETS) * len(QUESTIONS)
print("   %d calls, %d answers: %.2f s total" % (len(TICKETS), n_answers, loop_s))
print("   per ticket : %.1f ms" % (1000 * loop_s / len(TICKETS)))
print("   throughput : %.1f questions/second" % (n_answers / loop_s))

heading("one predict_batch() call for the queue")
t0 = time.perf_counter()
batch = agent.predict_batch(TICKETS, QUESTIONS, batch_size=4)
batch_s = time.perf_counter() - t0
print("   %d results: %.2f s total, %.1f questions/second" % (len(batch), batch_s, n_answers / batch_s))
agreement = sum(a["answers"]["department"]["choice"] == b["answers"]["department"]["choice"]
                for a, b in zip(single, batch))
print("   department agreement: %d/%d tickets" % (agreement, len(TICKETS)))
