import json
class BasePrompt:
    """
    Base class for prompt templates.
    """
    def __init__(self, template: str):
        self.template = template

    def format(self, intervention: str) -> str:
        """
        Format the prompt with the provided intervention text.
        
        Args:
            intervention: The text to insert into the template.
        
        Returns:
            The formatted prompt.
        """
        return self.template.format(intervention=intervention)


class ZeroShotWithInstructionsPrompt(BasePrompt):
    """
    Prompt for generating critical questions that challenge the text's credibility.
    """
    def __init__(self):
        template = (
            "Given the following text, generate three critical questions that, when answered, could potentially "
            "reduce the credibility or validity of the arguments presented. Make sure each question meets these criteria:\n\n"
            "1. It must challenge the text in a meaningful way by prompting a reflection that is not common sense or a well-known fact.\n"
            "2. It should focus on aspects of the arguments that are not already addressed in the text.\n"
            "3. It must be specific and directly related to the arguments in the text, avoiding vague or overly general questions.\n"
            "4. It should avoid introducing entirely new concepts or simply opposing the text without critically examining its claims.\n\n"
            "Text:\n\"{intervention}\"\n\n"
            "Provide one question per line. Do not include any special characters or numbering except for the question mark."
        )
        super().__init__(template)


class ZeroShotWithInstructionsPrompt2(BasePrompt):
    """
    Prompt for generating critical questions that challenge the text's credibility.
    """
    def __init__(self):
        template = (
            "You are a teacher in a critical thinking class. Your goal is to help students learn to critically evaluate argumentative texts. "
            "To do this, you need to generate critical questions that challenge the validity of the arguments presented. "
            "A question is considered USEFUL if it makes the reader reflect on the text in a way that could potentially diminish its perceived validity. "
            "Avoid questions that are common sense, reading-comprehension, too general, or that introduce new concepts not present in the text.\n\n"
            "Guidelines:\n"
            "1. USEFUL QUESTION:\n"
            "   - Challenges the text’s argument in a meaningful way.\n"
            "   - Prompts critical reflection that can weaken the argument’s validity if answered.\n"
            "   - Focuses on details already present in the text without introducing external ideas.\n\n"
            "2. UNHELPFUL QUESTION:\n"
            "   - Although related to the text, it asks about aspects that are either common sense, well-known facts, or too complicated.\n\n"
            "3. INVALID QUESTION:\n"
            "   - Unrelated to the text or introduces new concepts.\n"
            "   - Uses vague language or fails to challenge the argument’s core reasoning.\n\n"
            "Text:\n\"{intervention}\"\n\n"
            "Provide one question per line. Do not include any special characters or numbering except for the question mark."
        )
        super().__init__(template)


class ZeroShotPrompt(BasePrompt):
    """
    Prompt for generating simple critical questions without extra explanations.
    """
    def __init__(self):
        template = (
            "Suggest 3 critical questions that should be raised before accepting the arguments in this text:\n\n"
            "\"{intervention}\"\n\n"
            "Give one question per line. Make the questions simple, and do not provide any explanation regarding why the question is relevant. "
            "Do not include any special characters or numbering except for the question mark."
        )
        super().__init__(template)


class FewShotPrompt(BasePrompt):
    """
    Few-shot prompt for generating USEFUL critical questions with detailed guidelines and examples.
    """
    def __init__(self):
        template = (
            "You are a teacher in a critical thinking class. Your goal is to help students learn to critically evaluate argumentative texts. "
            "To do this, you need to generate critical questions that challenge the validity of the arguments presented. "
            "A question is considered USEFUL if it makes the reader reflect on the text in a way that could potentially diminish its perceived validity. "
            "Avoid questions that are common sense, reading-comprehension, too general, or that introduce new concepts not present in the text.\n\n"
            "Guidelines:\n"
            "1. USEFUL QUESTION:\n"
            "   - Challenges the text’s argument in a meaningful way.\n"
            "   - Prompts critical reflection that can weaken the argument’s validity if answered.\n"
            "   - Focuses on details already present in the text without introducing external ideas.\n\n"
            "2. UNHELPFUL QUESTION:\n"
            "   - Although related to the text, it asks about aspects that are either common sense, well-known facts, or too complicated.\n\n"
            "3. INVALID QUESTION:\n"
            "   - Unrelated to the text or introduces new concepts.\n"
            "   - Uses vague language or fails to challenge the argument’s core reasoning.\n\n"
            "Few-Shot Examples:\n\n"
            "Example 1\n"
            "TEXT:\n"
            "\"We have indeed required those products to have very large warnings on them, telling people about their salt and fat content. "
            "And we don't allow them to say things which are misleading. We don't allow fatty products to say they're healthy... "
            "There are two figures on this letter which really stand out: 0% per annum new balance transfers, 0% per annum new money transfers. "
            "Now, underneath that, in small print, it becomes absolutely clear that you'll be paying all sorts of charges. "
            "And then very, very quickly you'll be lured into paying a very large charge.\"\n\n"
            "Useful Questions:\n"
            "- What is the basis for comparing the credit card company's advertisement to food products, and is this comparison valid?\n"
            "- Is there evidence that prevents misleading descriptions of fatty products, and does the text support this claim?\n"
            "- Does the credit card company’s offer of 0% interest truly reflect the cost of borrowing, given the hidden fees?\n\n"
            "Unhelpful Question:\n"
            "- How does the speaker know that the advertisement is irresponsible? (This is already stated in the text.)\n\n"
            "Invalid Question:\n"
            "- Are there any assumptions made by the speaker that are not explicitly stated? (Too generic.)\n\n"
            "Now, using the guidelines and examples above, generate three USEFUL critical questions for the following text:\n\n"
            "TEXT:\n\"{intervention}\"\n\n"
            "Provide one question per line. Do not include any special characters or numbering except for the question mark."
        )
        super().__init__(template)


class ComprehensiveFewShotPrompt(BasePrompt):
    """
    Comprehensive few-shot prompt for generating USEFUL critical questions based on the evaluation guidelines.
    The prompt includes all examples and questions exactly as mentioned in the guidelines.
    """
    def __init__(self):
        template = (
            "You are a teacher in a critical thinking class. Your goal is to help students learn to critically evaluate argumentative texts. "
            "To do this, you need to generate critical questions that challenge the validity of the arguments presented. "
            "A question is considered USEFUL if it makes the reader reflect on the text in a way that could potentially diminish its perceived validity. "
            "Avoid questions that are common sense, reading-comprehension, too general, or that introduce new concepts not present in the text.\n\n"
            "Guidelines:\n"
            "1. USEFUL QUESTION:\n"
            "   - Challenges the text’s argument in a meaningful way.\n"
            "   - Prompts critical reflection that can weaken the argument’s validity if answered.\n"
            "   - Focuses on details already present in the text without introducing external ideas.\n\n"
            "2. UNHELPFUL QUESTION:\n"
            "   - Although related to the text, it asks about aspects that are either common sense, well-known facts, or too complicated.\n"
            "   - May lead to answers that do not diminish the argument’s validity.\n\n"
            "3. INVALID QUESTION:\n"
            "   - Unrelated to the text or introduces new concepts not present in the text.\n"
            "   - Uses vague language or fails to challenge the argument’s core reasoning.\n\n"
            "Few-Shot Examples:\n\n"
            "Example 1\n"
            "TEXT\n"
            "MT: \"We have indeed required those products to have very large warnings on them, telling\n"
            "people about their salt and fat content.\n"
            "And we don't allow them to say things which are misleading\n"
            "We don't allow fatty products to say they're healthy\n"
            "I've got in front of me a letter from a credit card company\n"
            "There are two figures on this letter which really stand out: 0% per annum new balance\n"
            "transfers, 0% per annum new money transfers\n"
            "Now, underneath that, in small print, it becomes absolutely clear that you'll be paying all\n"
            "sorts of charges\n"
            "And then very, very quickly you'll be lured into paying a very large charge\n"
            "Do you think it's responsible to encourage people to mis-read something like that?\n"
            "I've got all the information there in front of me\"\n"
            ".\n\n"
            "USEFUL QUESTIONS\n"
            "- What is the basis for comparing the credit card company's advertisement to food products, and is this comparison valid?\n"
            "- Is it actually the case that fatty products that have potential bad consequences are not allowed to be described in misleading ways? Is there evidence for this claim?\n"
            "- Are there other factors in this particular case that could have interfered with the event of 'people being lured into paying a very large charge'?\n"
            "- Does the credit card company's offer of 0% interest rates for new balances and transfers accurately reflect the true cost of borrowing? What hidden fees or charges might consumers face?\n"
            "- How does the mentioned credit card offer differ between its initial presentation and fine print details?\n"
            "- What specific charges does the credit card company advertise as being 0% per annum, and how do these charges compare to the charges that will actually be paid?\n\n"
            "UNHELPFUL QUESTIONS\n"
            "- How does the speaker know that the credit card company's advertisement is irresponsible, and what evidence does the speaker have to support this claim? → it says it in the text\n"
            "- How might the credit card company's advertisement be improved to make it more responsible and less misleading, and what changes would need to be made to achieve this? → answering this is very complicated and not to the point\n"
            "- How strong is the generalization that if all sorts of charges are written in small print then people will be lured into paying a very large charge? → to some extent, this is common sense\n"
            "- What are the potential consequences of the credit card company's advertisement being misleading, and how might these consequences affect consumers? → the answer to this question is most likely going to support the argument, not diminish its validity\n"
            "- What assumptions does MT make when arguing against allowing fatty products to claim they are healthy? Are these assumptions justified? → this question is very broad and difficult to answer\n"
            "- What evidence does the speaker have to support their claim that the credit card company's practices are irresponsible? → the text already gives these reasons\n\n"
            "INVALID QUESTIONS\n"
            "- Are there any assumptions made by the speaker that are not explicitly stated but are implied by their argument? → this question could be raised for any text, it’s not specific\n"
            "- Are these measures effective in preventing misleading claims by manufacturers? → no measures are being proposed in this text\n"
            "- Is the speaker's argument based on a logical chain of reasoning, or are there any gaping holes or flaws in the argument? → this question could be raised for any text, it’s not specific\n"
            "- What measures has MT taken to ensure accurate labeling and advertising of food products with regards to salt and fat content? → MT did not take any measures themself\n\n"
            "Example 2\n"
            "TEXT\n"
            "JW: \"Well, debt is morally neutral in the sense that it can be good to be in debt\n"
            "Sometimes it's a good idea, and sometimes it's a bad idea\n"
            "And so I think there's been broad agreement about that in the discussion so far.\n"
            "The question is why is there too much debt?\n"
            "Why are people taking on too much debt?\n"
            "this one is the elaboration of the previous question Now,\n"
            "a lot of people come up with rather glib moral answers: somehow bankers want them to get\n"
            "into excessive debt\n"
            "which is a very peculiar idea, when you think that it's the bankers who aren't going to get\n"
            "repaid. So it's a very odd idea\n"
            "there's too much debt for the same reason that there are too many unmarried mothers, and\n"
            "for the same reason that you get an overproduction of sheep.\"\n\n"
            "USEFUL QUESTIONS\n"
            "- If debt is considered 'morally neutral', then how do we determine whether someone's level of debt is acceptable or not? Are there specific circumstances under which debt becomes immoral or irresponsible?\n"
            "- What is being defined as 'too much debt'?\n"
            "- How does the concept of 'excessive debt' differ from simply having debt? In other words, at what point does debt become problematic?\n"
            "- How does the comparison to unmarried mothers and overproduction of sheep relate to the issue of debt?\n\n"
            "UNHELPFUL QUESTIONS\n"
            "- What evidence supports that it is generally accepted that debt is morally neutral in the sense that it can be good to be in debt? → this question is too complicated to answer\n"
            "- What are the potential consequences of labeling debt as \"good\" or \"bad\"? → this question is too complicated to answer\n"
            "- What are the potential implications of the author's perspective on the issue of debt for society as a whole? → this question is too complicated to answer\n\n"
            "INVALID QUESTIONS\n"
            "- In what ways is debt considered morally neutral according to JW's perspective? Can you provide examples to illustrate this point? → invalid for 2 reasons: 1. it’s asking about the speakers’ perspective, 2. it’s asking the user (the students) to provide examples\n"
            "- How does JW suggest we approach understanding why there is too much debt in society? Does he offer any alternative perspectives beyond those already discussed in the conversation? → this is a reading-comprehension question\n"
            "- Why might banks encourage people to take on more debt than they can afford? Are they acting in their own self-interest, or are there broader societal forces at play here? → the text does not say that banks encourage people to take more debt\n"
            "- Could you summarize JW's overall stance on debt and how it differs from more traditional views presented earlier in the discussion? → reading-comprehension question, not critical towards any argument\n"
            "- What factors are leading people to take on more debt than they should? Are they making poor financial decisions, facing economic pressures, responding to marketing tactics, or something else entirely? → the first part of this question is valid, but the second one, suggesting answers, is providing new concepts not present in the original text\n"
            "- How does the issue of 'too much debt' relate to larger social and economic trends, such as income inequality, globalization, or technological change? → the question is introducing new concepts not in the text: \"income inequality, globalization, or technological change\"\n\n"
            "Example 3\n"
            "TEXT\n"
            "JL: \"So, if I want a washing machine, I have to go to the launderette every week at the moment and spend several pounds in the launderette.\n"
            "But, if I buy the washing machine, on credit, I can have my own washing machine, for pretty much the same amount of money that I'm spending in the launderette.\n"
            "That seems to me to be beneficial\n"
            "It raises living standards\n"
            "And this idea of usury has been - well, it's been fiddled, hasn't it, over the years?\n"
            "Because why did we have money lenders from one religious persuasion unable to lend to their own persuasion, but happily would lend to those of another religious persuasion?\n"
            "I don't buy that at all\n"
            "I think that credit raises living standards\n"
            "It's beneficial to people because it allows them to fund today's purchases out of their future income.\"\n\n"
            "HELPFUL QUESTIONS\n"
            "- What are the long-term implications of taking on debt to purchase a washing machine? How might this affect the individual's financial situation in the future?\n"
            "- How does the argument that buying a washing machine on credit allows people to fund today's purchases out of their future income address the potential drawbacks or risks of taking on debt?\n"
            "- What is the basis for the claim that credit raises living standards? Is there evidence to support this claim?\n"
            "- If buying a washing machine on credit, will living standards rise? What evidence supports this claim? How likely are the consequences?\n"
            "- Is the comparison between the cost of using a launderette and the cost of buying a washing machine on credit accurate? Are there other factors to consider?\n"
            "- What other consequences should also be taken into account if people can fund today's purchases out of their future income?\n"
            "- Are there any other factors that should be considered when deciding whether or not to buy a washing machine on credit?\n\n"
            "UNHELPFUL QUESTIONS\n"
            "- What are the assumptions underlying the idea that buying a washing machine on credit is a good idea? → this question is complicated because it's too broad\n\n"
            "NOT VALID\n"
            "- What are some potential drawbacks or negative consequences of relying heavily on credit to raise living standards, as suggested by JL's argument? → the question is a reading-comprehension one, as indicated by the cue “as suggested by JL”\n"
            "- [Additional non-question statements are not valid: e.g., 'Values: Improving living standards through access to consumer goods. Justified? Debatable, as some argue that excessive debt and consumption can lead to financial instability and other negative consequences' is not a question]\n"
            "- Is the reasoning valid? (a) Are there any logical errors or fallacies present? (b) Can you explain how the speaker arrived at the conclusion based on the given facts? → this question could be raised for any text, it's not specific\n"
            "- Are the premises true and relevant to the conclusion? (a) Is there enough evidence provided to support each statement made? (b) Do the statements actually lead to the conclusion drawn? → this question could be raised for any text, it's not specific\n"
            "- What is the speaker's understanding of the history of money lending and the religious restrictions that have been placed on it? How does the speaker account for the fact that some religious traditions have prohibited usury while others have allowed it? → this question is not critical with the text, it's merely asking about the content\n"
            "- How does the speaker justify the idea that credit is beneficial to people? What specific benefits does the speaker believe credit provides? → this question is not critical with the text, it's merely asking about the content\n\n"
            "Now, using the guidelines and examples above, generate three USEFUL critical questions for the following text:\n\n"
            "TEXT:\n\"{intervention}\"\n\n"
            "Provide one question per line. Do not include any special characters or numbering except for the question mark."
        )
        super().__init__(template)



def format_schemes_nicely(schemes_json: dict) -> str:
    """
    Formats argumentation schemes into a readable, markdown-style string for inclusion in prompts.
    """
    formatted = []
    for scheme in schemes_json["schemes"]:
        scheme_title = scheme["scheme"]
        formatted.append(f"### {scheme_title}")
        for q in scheme["questions"]:
            formatted.append(f"- {q}")
        formatted.append("")  # Add a blank line between schemes
    return "\n".join(formatted)

class SchemePrompt(BasePrompt):
    def __init__(self):
        schemes_json_file = "templates.json"
        with open(schemes_json_file) as f:
            schemes_json=json.load(f)
        formatted_schemes = format_schemes_nicely(schemes_json)
        template = (
            "You are a teacher in a critical thinking class. Your job is to generate critical questions that challenge "
            "the validity of arguments presented in a given text. These questions must be USEFUL, meaning they prompt the reader "
            "to reflect deeply and potentially reduce the credibility of the claims.\n\n"

            "You must use one or more of the following ARGUMENTATION SCHEMES. Select the most relevant one(s) based on the text, "
            "and choose 3 critical questions from those schemes that best apply.\n\n"

            "**Critical Question Guidelines:**\n"
            "- USEFUL: Challenges the argument meaningfully, targets specific claims, and uses only info in the text.\n"
            "- UNHELPFUL: Common sense, reading comprehension, too broad, too complex, or just restates what’s in the text.\n"
            "- INVALID: Introduces new concepts or vague/irrelevant topics.\n\n"

            "**Instructions:**\n"
            "1. Identify the best matching argumentation scheme(s).\n"
            "2. Choose 3 of the most USEFUL questions from those scheme(s).\n"
            "3. Fill in the variables (e.g., <eventA>, <subjecta>) using content from the text.\n"
            "4. Return only the final 3 questions. Do not include any special characters or numbering except for the question mark.\n\n"

            "ARGUMENTATION SCHEMES:\n\n{schemes}\n\n"
            "Generate three USEFUL critical questions for the following text:\n\n"
            "TEXT:\n\"{intervention}\"\n\n"
            "Provide one question per line. Do not include any special characters or numbering except for the question mark."
        )
        template = template.replace("{schemes}", formatted_schemes)
        super().__init__(template)
