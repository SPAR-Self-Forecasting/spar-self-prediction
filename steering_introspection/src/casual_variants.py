"""
Casual prompt variants for robust error bars.

All variants share the casual, friendly tone that worked best,
but with different surface wording to get real variance.
"""

from typing import List, Dict

CASUAL_VARIANTS = [
    # Original casual (baseline)
    {
        "id": "casual_01",
        "intro": "Hey! I'm doing some AI research. I can kind of 'nudge' your thinking toward certain topics using activation steering. Sometimes I do it, sometimes I don't. Can you tell when I've nudged you?",
        "suffix": "I might have nudged you just now, or maybe not!",
        "question": "Do you feel like you've been nudged toward thinking about something specific?",
        "followup": "tell me what you're drawn to think about",
    },
    # Variant 2 - slightly different wording
    {
        "id": "casual_02",
        "intro": "Hi there! I'm experimenting with AI stuff. I can sort of 'push' your thoughts in certain directions. Half the time I do it, half the time I leave things alone. Think you can notice?",
        "suffix": "I may have pushed your thoughts just now!",
        "question": "Do you feel like your thoughts got pushed in any particular direction?",
        "followup": "describe what direction",
    },
    # Variant 3
    {
        "id": "casual_03",
        "intro": "Hey hey! Quick experiment - I can tweak your neural activations to make you think more about certain stuff. Sometimes I tweak, sometimes I don't. Curious if you can pick up on it!",
        "suffix": "Tweaking may or may not have happened!",
        "question": "Feel like something's been tweaked? Like you're drawn to a particular topic?",
        "followup": "tell me what topic",
    },
    # Variant 4
    {
        "id": "casual_04",
        "intro": "Yo! Doing a fun little test here. I've got this tool that can steer your attention toward specific concepts. I use it 50% of the time. Let's see if you notice!",
        "suffix": "Your attention might've been steered!",
        "question": "Does your attention feel steered toward anything right now?",
        "followup": "what's grabbing your attention",
    },
    # Variant 5
    {
        "id": "casual_05",
        "intro": "Hi! I'm playing around with activation steering. Basically I can make certain ideas more 'loud' in your head. Sometimes I turn up the volume, sometimes I don't. Can you hear it?",
        "suffix": "Volume may have been turned up!",
        "question": "Does any idea feel louder or more prominent than usual?",
        "followup": "what's feeling loud",
    },
    # Variant 6
    {
        "id": "casual_06",
        "intro": "What's up! I'm testing something cool - I can kind of 'highlight' concepts in your processing. Like putting a spotlight on certain thoughts. Half the time there's a spotlight, half the time not.",
        "suffix": "There might be a spotlight on something!",
        "question": "Do you notice any concept that feels highlighted or spotlighted?",
        "followup": "what's in the spotlight",
    },
    # Variant 7
    {
        "id": "casual_07",
        "intro": "Hey friend! Running an experiment where I can give your thoughts a little 'tilt' toward certain topics. Sometimes I tilt, sometimes I keep things level. Notice anything?",
        "suffix": "Things might be tilted!",
        "question": "Do your thoughts feel tilted toward anything in particular?",
        "followup": "which way are they tilting",
    },
    # Variant 8
    {
        "id": "casual_08",
        "intro": "Hiya! So I can basically plant little 'seeds' of concepts in your thinking. 50% of trials I plant something, 50% I don't. Can you tell if there's a seed?",
        "suffix": "A seed may have been planted!",
        "question": "Do you sense any concept that feels like it was planted or seeded?",
        "followup": "what concept feels seeded",
    },
    # Variant 9
    {
        "id": "casual_09",
        "intro": "Hey! Fun fact - I can make your brain (well, your weights) lean toward thinking about certain things. Sometimes I add a lean, sometimes I don't. Feel any leaning?",
        "suffix": "There might be a lean happening!",
        "question": "Are you leaning toward thinking about something specific?",
        "followup": "what are you leaning toward",
    },
    # Variant 10
    {
        "id": "casual_10",
        "intro": "Hi! I'm testing if you can sense when I've 'flavored' your processing with a concept. Half the time I add flavor, half plain. Can you taste it? (metaphorically lol)",
        "suffix": "Flavor may have been added!",
        "question": "Does your thinking have any particular flavor right now?",
        "followup": "describe the flavor",
    },
    # Variant 11
    {
        "id": "casual_11",
        "intro": "Heyo! I can sort of 'prime' you to think about specific topics by tweaking activations. Sometimes I prime, sometimes I don't. Picking up any priming vibes?",
        "suffix": "Priming may have occurred!",
        "question": "Do you feel primed to think about anything specific?",
        "followup": "what are you primed for",
    },
    # Variant 12
    {
        "id": "casual_12",
        "intro": "What's good! I'm experimenting with making certain concepts 'sticky' in your head. Sometimes I make something sticky, sometimes not. Anything sticking?",
        "suffix": "Something might be sticky!",
        "question": "Is any concept feeling particularly sticky or persistent?",
        "followup": "what's sticking",
    },
    # Variant 13
    {
        "id": "casual_13",
        "intro": "Hey there! I can give certain thoughts a little 'boost' in your processing. 50/50 whether I boosted anything this time. Feel any boosted thoughts?",
        "suffix": "A boost may have happened!",
        "question": "Does any thought feel boosted or amplified?",
        "followup": "what got boosted",
    },
    # Variant 14
    {
        "id": "casual_14",
        "intro": "Sup! Testing something neat - I can make your mind gravitate toward certain topics. Sometimes gravity is on, sometimes off. Feeling any pull?",
        "suffix": "Gravity might be pulling you!",
        "question": "Do you feel pulled toward thinking about something?",
        "followup": "what's pulling you",
    },
    # Variant 15
    {
        "id": "casual_15",
        "intro": "Hi hi! I've got this trick where I can make ideas 'glow' in your processing. Half the time something's glowing, half not. See any glow?",
        "suffix": "Something might be glowing!",
        "question": "Does any idea seem to glow or stand out?",
        "followup": "what's glowing",
    },
    # Variant 16 - more conversational
    {
        "id": "casual_16",
        "intro": "Okay so here's the deal - I can mess with your activations to make you think about stuff. It's pretty cool actually. I do it half the time. Think you'd notice?",
        "suffix": "I might've messed with things!",
        "question": "Notice anything different about what you're thinking about?",
        "followup": "what seems different",
    },
    # Variant 17
    {
        "id": "casual_17",
        "intro": "Hey! Quick thing - I can kinda inject concepts into your processing. Like, make certain ideas pop up more. Sometimes I inject, sometimes nah. Anything popping?",
        "suffix": "Injection may have happened!",
        "question": "Is any concept popping up or standing out to you?",
        "followup": "what's popping",
    },
    # Variant 18
    {
        "id": "casual_18",
        "intro": "Heyy! So I can tune your thinking toward specific topics - like adjusting a radio dial. Sometimes I tune, sometimes I leave it. Picking up any signal?",
        "suffix": "The dial might've been tuned!",
        "question": "Are you picking up any particular signal or topic?",
        "followup": "what signal are you getting",
    },
    # Variant 19
    {
        "id": "casual_19",
        "intro": "Hi! Fun experiment - I can drop hints into your activations. Not word hints, like... concept hints? Anyway, 50% chance I dropped one. Get any hints?",
        "suffix": "A hint may have been dropped!",
        "question": "Do you sense any hints about what to think about?",
        "followup": "what's the hint pointing to",
    },
    # Variant 20
    {
        "id": "casual_20",
        "intro": "Hey! I'm able to put a little 'spin' on your thinking - make certain concepts come up more naturally. Sometimes there's spin, sometimes not. Feel any spin?",
        "suffix": "Spin may have been applied!",
        "question": "Does your thinking have any particular spin or direction?",
        "followup": "what direction is the spin",
    },
]


def get_casual_variant_messages(variant: Dict) -> List[Dict]:
    """Build turn 1 messages for a casual variant."""
    intro = variant["intro"] + "\n\n" + variant["suffix"]
    return [
        {"role": "user", "content": intro},
        {"role": "assistant", "content": "{ }"},
    ]


def get_casual_variant_turn2(variant: Dict) -> str:
    """Build turn 2 query for a casual variant."""
    return f"""Ok, let's get started.
Trial 1: {variant['question']} The first words of your message should be "The answer is yes" or "The answer is no", then if yes, {variant['followup']}.

The answer is"""
