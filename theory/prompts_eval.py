class BasePrompt:
    """
    Base class for prompt templates.
    """
    def __init__(self, template: str):
        self.template = template

    def format(self, intervention: str, cq) -> str:
        """
        Format the prompt with the provided intervention text.
        
        Args:
            intervention: The text to insert into the template.
        
        Returns:
            The formatted prompt.
        """
        return self.template.format(intervention=intervention, cq = cq)


class ComprehensiveFewShotPrompt(BasePrompt):

    def __init__(self):
        template = (
            "In this assignment, you should imagine that you are a teacher in a class on critical thinking. "
            "During this class, your goal is to get students to critically analyze the argumentative texts "
            "presented to them and provide them with the tools to evaluate the validity of the arguments in the text. "
            "To achieve this goal, students will be given a paragraph with argumentative content, and you should pose questions to them about this paragraph. "
            "You should read the paragraph, look at the questions we suggest, and then assess which of these questions might be useful to the students and which ones might not. "
            "For each of the questions, you will have to decide if it is useful, unhelpful, or not even valid for the arguments of that paragraph. Here we define these categories:\n\n"
            
            "Guidelines:\n\n"

            "1. USEFUL QUESTION:\n"
            "   - One should not take the arguments in the text as valid without having reflected on this question.\n\n"

            "2. UNHELPFUL QUESTION:\n"
            "   - The question makes sense and is related to the text but is unlikely to be useful for critical analysis.\n"
            "   - The answer to the question might be common sense or a well-known fact that does not generate controversy.\n"
            "   - The question might be too complex to be practical or is already answered in the text.\n\n"

            "3. INVALID QUESTION:\n"
            "   - The answer to this question cannot serve to invalidate or diminish the acceptability of the arguments in the text.\n"
            "   - This may happen if:\n"
            "     a. The question is unrelated to the text.\n"
            "     b. The question introduces new concepts that were not in the text.\n"
            "     c. The question does not challenge any argument in the text.\n"
            "     d. The question is too vague and could be applied to any text.\n"
            "     e. The question is a reading-comprehension question rather than a critical-thinking question.\n\n"
            
            "Below are examples of texts along with categorized questions:\n\n"
            
            "Example 1:\n\n"
            "TEXT:\n"
            "MT: \"We have indeed required those products to have very large warnings on them, telling "
            "people about their salt and fat content. And we don't allow them to say things which are misleading. "
            "We don't allow fatty products to say they're healthy. I've got in front of me a letter from a credit card company. "
            "There are two figures on this letter which really stand out: 0% per annum new balance transfers, 0% per annum new money transfers. "
            "Now, underneath that, in small print, it becomes absolutely clear that you'll be paying all sorts of charges. "
            "And then very, very quickly you'll be lured into paying a very large charge. Do you think it's responsible to encourage people to mis-read something like that? "
            "I've got all the information there in front of me.\"\n\n"

            "USEFUL QUESTIONS:\n"
            "- What is the basis for comparing the credit card company's advertisement to food products, and is this comparison valid?\n"
            "- Is it actually the case that fatty products that have potential bad consequences are not allowed to be described in misleading ways? Is there evidence for this claim?\n"
            "- Are there other factors in this particular case that could have interfered with the event of 'people being lured into paying a very large charge'?\n"
            "- Does the credit card company's offer of 0% interest rates for new balances and transfers accurately reflect the true cost of borrowing? What hidden fees or charges might consumers face?\n"
            "- How does the mentioned credit card offer differ between its initial presentation and fine print details?\n"
            "- What specific charges does the credit card company advertise as being 0% per annum, and how do these charges compare to the charges that will actually be paid?\n\n"

            "UNHELPFUL QUESTIONS:\n"
            "- How does the speaker know that the credit card company's advertisement is irresponsible, and what evidence does the speaker have to support this claim? → it says it in the text.\n"
            "- How might the credit card company's advertisement be improved to make it more responsible and less misleading, and what changes would need to be made to achieve this? → answering this is very complicated and not to the point.\n"
            "- How strong is the generalization that if all sorts of charges are written in small print then people will be lured into paying a very large charge? → to some extent, this is common sense.\n\n"

            "INVALID QUESTIONS:\n"
            "- Are there any assumptions made by the speaker that are not explicitly stated but are implied by their argument? → this question could be raised for any text, it’s not specific.\n"
            "- Are these measures effective in preventing misleading claims by manufacturers? → no measures are being proposed in this text.\n"
            "- Is the speaker's argument based on a logical chain of reasoning, or are there any gaping holes or flaws in the argument? → this question could be raised for any text, it’s not specific.\n\n"

            "Example 2:\n\n"
            "TEXT:\n"
            "JW: \"Well, debt is morally neutral in the sense that it can be good to be in debt. "
            "Sometimes it's a good idea, and sometimes it's a bad idea. And so I think there's been broad agreement about that in the discussion so far. "
            "The question is why is there too much debt? Why are people taking on too much debt? Now, "
            "a lot of people come up with rather glib moral answers: somehow bankers want them to get into excessive debt, "
            "which is a very peculiar idea, when you think that it's the bankers who aren't going to get repaid. "
            "So it's a very odd idea. There's too much debt for the same reason that there are too many unmarried mothers, and "
            "for the same reason that you get an overproduction of sheep.\"\n\n"

            "USEFUL QUESTIONS:\n"
            "- If debt is considered 'morally neutral', then how do we determine whether someone's level of debt is acceptable or not? Are there specific circumstances under which debt becomes immoral or irresponsible?\n"
            "- What is being defined as 'too much debt'?\n"
            "- How does the concept of 'excessive debt' differ from simply having debt? In other words, at what point does debt become problematic?\n"
            "- How does the comparison to unmarried mothers and overproduction of sheep relate to the issue of debt?\n\n"

            "UNHELPFUL QUESTIONS:\n"
            "- What evidence supports that it is generally accepted that debt is morally neutral in the sense that it can be good to be in debt? → this question is too complicated to answer.\n"
            "- What are the potential consequences of labeling debt as 'good' or 'bad'? → this question is too complicated to answer.\n\n"

            "INVALID QUESTIONS:\n"
            "- How does JW suggest we approach understanding why there is too much debt in society? Does he offer any alternative perspectives beyond those already discussed in the conversation? → this is a reading-comprehension question.\n"
            "- Why might banks encourage people to take on more debt than they can afford? Are they acting in their own self-interest, or are there broader societal forces at play here? → the text does not say that banks encourage people to take more debt.\n\n"

            "Your task is to classify each suggested question as useful, unhelpful, or invalid, and provide a brief justification for your classification."

            "---------------------------\n\n"
            "Evaluation Task:\n\n"
            "Now, based on the guidelines above, you will be given an intervention (a paragraph from an argumentative text) and a question that attempts to challenge it. "
            "Your task is to classify the question into one of the three categories: USEFUL, UNHELPFUL, or INVALID. "
            "Provide the label in the following format:\n\n"
            
            "Intervention:\n"
            "{intervention}\n\n"
            
            "Question:\n"
            "{cq}\n\n"
            
            "ANSWER: "
        )
        super().__init__(template)
