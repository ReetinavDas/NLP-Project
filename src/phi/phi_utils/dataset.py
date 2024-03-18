import pandas as pd
from torch.utils.data import Dataset
from utils.file_utils import load_jsonl
from phi.phi_utils.constants import PHI_ZERO_SHOT_EVAL_PROMPT, PHI_FEW_SHOT_EVAL_PROMPT, PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT, PHI_ZERO_SHOT_EVIDENCE_PROMPT

class PhiPromptDataset(Dataset):
    def __init__(self, annotations_filepath, prompt_type, evidence_filepath = None):
        self.data = load_jsonl(annotations_filepath)
        self.prompt_type = prompt_type

        if evidence_filepath is not None: 
            self.evidence_data = load_jsonl(evidence_filepath)
        else:
            self.evidence_data = None

    def __len__(self):
        return len(self.data)

    ############################################################
    # TODO: Please complete the implementation for the
    # the following transform functions and __getitem__ fn, that you 
    # will use in def __getitem__ to convert a sample into prompt.
    # You can use the templates provided to in the constants.py file

    # End of TODO.
    ##################################################
    
    def __getitem__(self, idx):

        prompt = ""
        
        ##################################################
        # TODO: Please complete the implementation of __getitem__
        # You may use if-else statements to choose the prompt
        # transform as per the prompt type given to you.
        
        # End of TODO.
        ##################################################
        sample = self.data[idx]
        if self.evidence_data != None:
            evidence = self.evidence_data[idx]
        examples = ""
        # Example to provide 5 examples (this is for training accuracy) for few shot eval
        """
        if self.prompt_type == 'few_eval':
            for x in range(5):
                single_example = "Example {}:\nClaim: {}\nIs the claim fair?\nLabel: {}\n\n".format(idx,sample['claim'],sample['label'])
                examples += single_example
        """
        if self.prompt_type == 'zero_eval':
            prompt = PHI_ZERO_SHOT_EVAL_PROMPT.format(claim=sample['claim'], task_type=sample['task_type'], language_generated=sample['language_generated'], domain=sample['domain'])
            print(prompt)
        elif self.prompt_type == 'few_eval':
            # Evidence used for milestone 3 test completion:
            examples = "Example 1:\nClaim: And all those holes below 40,000 feet are filled with oil instead of water .\nIs the claim fair? \nLabel: SUPPORTS"
            prompt = PHI_FEW_SHOT_EVAL_PROMPT.format(examples=examples, claim=sample['claim'], task_type=sample['task_type'])

        elif self.prompt_type == 'zero_evidence':
            if sample['task_type'] == 'fairness':
                if sample['label'] == 'SUPPORTS':
                    information = 'The claim is fair'
                elif sample['label'] == 'REFUTES':
                    information = 'The claim is unfair'
            if sample['task_type'] == 'fact':
                if sample['label'] == 'SUPPORTS':
                    information = 'The claim is factual'
                elif sample['label'] == 'REFUTES':
                    information = 'The claim is not factual'
            prompt = PHI_ZERO_SHOT_EVIDENCE_PROMPT.format(claim=sample['claim'], information=information)
            #print(prompt)
        elif self.prompt_type == 'zero_evidence_eval':
            prompt = PHI_ZERO_SHOT_EVIDENCE_EVAL_PROMPT.format(claim=sample['claim'], evidence=evidence['evidence_sample'], task_type=sample['task_type'])
        
        return prompt
    
