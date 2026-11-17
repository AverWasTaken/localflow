"""Command mode questions (localflow/ask.py): question vs command routing, exact local answers, answer tidying."""
import sys
import unittest
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from localflow import ask  # noqa: E402

COMMANDS = ["Could you just shorten this?", "Can you help me rewrite this?", "Would you mind fixing the grammar?",
            "Can we make this friendlier?", "Do this in bullet points", "Can you make this more formal?",
            "Make it formal", "Translate to German", "make it formal", "Make this all uppercase.",
            "Write a reply saying I'll be late", "Can I get a summary of this?", "Please make this shorter",
            "Can you also make this shorter?", "Hey, can you fix the typos?", "Summarize this", "Do it again",
            "Put this in quotes", "12 times 7", "Type what's the capital of France", "Insert today's date",
            "Will you translate this to Spanish?", "Go ahead and delete the last sentence", "", "um",
            # "can you <verb>" with a verb that isn't asking for information: a request, stays a command
            "Can you finish this sentence", "Can you update this to say Friday instead", "Can you sort these alphabetically",
            "Can you combine these two sentences", "Can you condense this", "Can you create a bulleted summary of this",
            "Can you take this and make it shorter", "Can we meet at 5?", "Do a spell check",
            "Do the same for the second paragraph", "Do an outline of this", "Do these in bullet points",
            "Do those in French", "Do these as a numbered list", "Do the rest", "Can you tell him I'll be late",
            "Could you tell the client we need more time", "Will you remind John about the deadline",
            "Can't you make this shorter", "Don't change this", "Help me with this email", "Help me fix this",
            "Ask them to reschedule", "Ask John if he can make it", "Explain to the client that the shipment is late",
            "Can you explain to the team why we moved the date"]
QUESTIONS = ["How many hours is it from 7:30am to 5pm?", "What's 15% of 240?", "What does this mean?",
             "Is this grammatically correct?", "Can you explain this", "Can you explain this?",
             "Who wrote Pride and Prejudice?", "Ask what the capital of France is", "Does this sound rude?",
             "Do you know what time it is in Tokyo?", "Can you tell me what this does?",
             "Can you explain the change?", "Question, is it going to rain?", "Tell me a fun fact",
             "Hey what's the capital of Peru?", "Will this work in Excel?", "Calculate 12 times 7",
             "Do these sentences sound right?", "Would you mind explaining this?", "whats the weather like",
             "So how long does pasta take to cook?", "Why is the sky blue", "Define serendipity",
             "Ask: make this formal",
             # a filler word in front doesn't turn a question into a command
             "Hey, how do I fix this?", "So how do I convert celsius to fahrenheit", "So what do you think of this",
             "Okay, do you know the capital of France", "So what type of file is this", "Um what does this mean",
             "Can you also explain this?", "Could you check if this is correct?", "Can you tell me how to make pasta",
             "Would you know who wrote this?", "Do the numbers add up?", "Do the trains run on Sunday?",
             "Do the dates in this match?", "Doesn't this sound rude?", "Shouldn't this be capitalized?",
             "Didn't I already say this", "Wasn't this fixed", "Can you tell us the deadline",
             "Can you answer this question: how many hours is it from 7:30am to 5pm?", "Can you count the words in this",
             "Help me understand this", "Can you help me understand this", "Explain to me how this works",
             "Ask, how far is the moon"]
NOW = datetime(2026, 11, 10, 15, 42)


class IsQuestion(unittest.TestCase):
    def test_commands_stay_commands(self):
        for c in COMMANDS:
            self.assertFalse(ask.is_question(c), c)

    def test_questions(self):
        for q in QUESTIONS:
            self.assertTrue(ask.is_question(q), q)

    def test_no_question_mark_rule(self):
        self.assertFalse(ask.is_question("Shorter?"))
        self.assertFalse(ask.is_question("Make it formal?"))

    def test_question_text_drops_override_word(self):
        self.assertEqual(ask.question_text("Ask what the capital of France is"), "What the capital of France is")
        self.assertEqual(ask.question_text("Question, is it going to rain?"), "Is it going to rain?")
        self.assertEqual(ask.question_text("ask: make this formal"), "Make this formal")
        self.assertEqual(ask.question_text("What does this mean?"), "What does this mean?")
        self.assertEqual(ask.question_text("Asking for a friend"), "Asking for a friend")

    def test_references_text(self):
        for q in ("What does this mean?", "Is that right", "Summarize the selected part", "what is the text about",
                  "Do I need a comma here?"):
            self.assertTrue(ask.references_text(q), q)
        for q in ("Is the grammar correct?", "Any typos?", "Does the tone sound okay?"):
            self.assertTrue(ask.references_text(q), q)
        for q in ("Who wrote Pride and Prejudice?", "What's the capital of Peru?", "What time is it"):
            self.assertFalse(ask.references_text(q), q)


class QuickAnswer(unittest.TestCase):
    def q(self, s):
        return ask.quick_answer(s, NOW)

    def test_time_spans(self):
        cases = {
            "How many hours is it from 7:30am to 5pm?": "9 hours 30 minutes (9.5 hours)",
            "How many hours from 7:30 a.m. to 5 p.m.?": "9 hours 30 minutes (9.5 hours)",
            "How many hours is it from 7.30AM to 5PM?": "9 hours 30 minutes (9.5 hours)",
            "how many hours from 9 to 5": "8 hours",
            "How long is it from 10pm to 6am?": "8 hours",
            "How much time is there from noon to 1:15pm?": "1 hour 15 minutes (1.25 hours)",
            "How many hours from 7:30am to 5?": "9 hours 30 minutes (9.5 hours)",       # missing meridiem
            "How many hours from 9 to 5pm?": "8 hours",
            "How many hours from 10pm to 6?": "8 hours",                                  # overnight
            "how many hours between 7:30 and 17:30": "10 hours",
            "how many hours between 730am and 530pm": "10 hours",
            "How many minutes from 9 to 10:30?": "90 minutes (1 hour 30 minutes)",
            "How many hours from 9:20 to 5": "7 hours 40 minutes",
            "how many hours from 9am to 9am": "24 hours",
            "how long from 11:45pm until midnight": "15 minutes",
            "Hey, how many hours is it from 8 a.m. to 4:45 p.m.?": "8 hours 45 minutes (8.75 hours)",
            "Can you tell me how many hours from 9 to 5?": "8 hours",
            "Can you tell me how many hours it is from 7:30am to 5pm": "9 hours 30 minutes (9.5 hours)",
            "How many hours are in between 7:30am and 5pm": "9 hours 30 minutes (9.5 hours)",
            "How many hours is 7:30am to 5pm": "9 hours 30 minutes (9.5 hours)",
        }
        for s, want in cases.items():
            self.assertEqual(self.q(s), want, s)

    def test_arithmetic(self):
        cases = {
            "What's 15% of 240?": "36", "What is 12 times 7?": "84", "What is 144 divided by 12?": "12",
            "What's 3.5 + 2?": "5.5", "what is 1234 times 5678": "7,006,652", "What's 1 divided by 3?": "0.333333",
            "What's (2 + 3) * 4?": "20", "What is 1,000 plus 25?": "1025", "What's 2.50 times 4?": "10",
            "What's 7 x 8?": "56", "what is 0.1 plus 0.2": "0.3", "What's -5 + 3?": "-2",
            "How much is 20 percent of 50?": "10", "Calculate 1 over 8": "0.125", "compute 10 minus 15.5": "-5.5",
            "What is 17 percent of 3,450 plus 12?": "598.5", "What is 100000 divided by 4?": "25,000",
            "Can you calculate 17% of 3450 plus 12?": "598.5", "Could you please compute 6 times 7": "42",
            "Calculate 144/12": "12", "How much is 144/12 + 3": "15",
        }
        for s, want in cases.items():
            self.assertEqual(self.q(s), want, s)

    def test_none_when_not_the_whole_question(self):
        for s in ("how many integers from 3 to 9", "what's a good number between 5 and 10",
                  "how many hours from 9 to 5 minus a 30 minute lunch", "12 times 7", "what is 2",
                  "what's the capital of France", "what's 10 divided by 0", "what is 2 to the power of 8",
                  "what's 2 ** 8", "What's 5 + 3 in binary?", "how many hours from 13pm to 2", "what is 5 +",
                  "how many hours from 9:75 to 5", "", "What does this mean?", "How long from 2010 to 2024",
                  "How long is it from 2019 to 2023?", "How much time between 2015 and 2020", "what's 9/11",
                  "what is 24/7", "what's 24-7", "what is 2026-11-10"):
            self.assertIsNone(self.q(s), s)


class TidyAnswer(unittest.TestCase):
    def test_tidy(self):
        t = ask.tidy_answer
        self.assertEqual(t("<think></think>**Answer:** The capital is *Paris*."), "The capital is Paris.")
        self.assertEqual(t("<think></think>A: 3 * 4 = 12"), "3 * 4 = 12")
        self.assertEqual(t("<think>hm</think>Canberra"), "Canberra")
        self.assertEqual(t("<question>x</question> `Jane` __Austen__"), "x Jane Austen")
        self.assertEqual(t("- one\n- two\n\n  three  "), "one two three")
        self.assertEqual(t("```\n42\n```"), "42")
        self.assertEqual(t("<think></think>  "), "")
        self.assertEqual(t("<think>never finished"), "")
        self.assertEqual(t(""), "")


if __name__ == "__main__":
    unittest.main()
