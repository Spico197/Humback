from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template

register_conv_template(
    Conversation(
        name="vicuna_v1.1_reverse",
        system_message=(
            "This is a chat between a curious user and a helpful artificial intelligence assistant. "
            "Given the assistant's reponse, please predict the user's instruction."
        ),
        roles=("RESPONSE", "INSTRUCTION"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)
