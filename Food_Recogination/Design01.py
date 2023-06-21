import requests
import streamlit as st
import tensorflow as tf
import json
import streamlit_lottie
from  streamlit_lottie import st_lottie
import toml
st.set_page_config(page_title="Hello Foodie!!",page_icon=":tada:",layout="wide")

# config = toml.load('C:\\Users\\Lenovo\\PycharmProjects\\pythonProject6\\template\\config.toml')
# db_host = config['database']['host']
# st.write(f"Database host: {db_host}")
page_bg_img="""
<style>
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)
def load_model():
    model = tf.keras.models.load_model("C:\\Users\\Lenovo\\Untitled Folder 4\\my_model3.hdf5")
    return model
model = load_model()

from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    size = (256, 256)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

class_names = ['Briyani', 'Dhokla', 'Dosa', 'Gulab_Jamun', 'Idli', 'Palak_Panner', 'ButterPaneerMasala', 'Poha', 'Vada', 'Samosa','VadaPav']


def load_lottieurl(url):
    r= requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()

lottie_coding= load_lottieurl ("https://assets4.lottiefiles.com/packages/lf20_YnsM0o.json")
lottie_coding2= load_lottieurl ("https://assets3.lottiefiles.com/temp/lf20_nXwOJj.json")
lottie_coding3= load_lottieurl ("https://assets3.lottiefiles.com/packages/lf20_TmewUx.json")
lottie_coding4= load_lottieurl ("https://assets10.lottiefiles.com/packages/lf20_GxMZME.json")


def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>",unsafe_allow_html=True)

load_css("Css_file")

#HEader section
with st.container():
    st.subheader("Hello, Welcome to the Food Detection model!")
    st.title("Food Detection and Recipe Recomendation")
    st.write("Let's Begin!!")


with st.container():
    st.write("---")
    left_column, right_column= st.columns(2)
    with left_column:
        st.header("Our Aim:")
        st.write("##")
        st.write("""
        
        Welcome to our food detection website! Our website is designed to help you identify different types of food through image recognition technology. We have trained a deep convolutional neural network (CNN) model to classify 11 different food types based on the input image.
        With our website, you can upload an image of food and get an accurate prediction of the type of food it is. Our model has been trained on a large dataset of food images and is capable of recognizing a wide range of food types."""
        )

    with right_column:
        st_lottie(lottie_coding, height= 300, key="Food")


with st.container():
    recipes = {
        'Briyani': """
        Recipe for Briyani: ...
    Heat ghee or oil in a large pot or pan over medium heat.
    Add the sliced onions and sauté until golden brown.
    Add the chopped tomatoes and cook until they turn soft.
    Add the mixed vegetables and sauté for a few minutes.
    Add the yogurt, turmeric powder, red chili powder, biryani masala, and salt. Mix well.
    Add the soaked and drained Basmati rice to the pot and gently mix with the vegetables and spices.
    Add 3 cups of water and bring it to a boil.
    Reduce the heat to low, cover the pot with a tight-fitting lid, and cook for about 15-20 minutes until the rice is cooked and the flavors are well absorbed.
    Meanwhile, heat a small pan and dry roast the saffron strands for a minute.
    Add the saffron strands to the milk and let it steep for a few minutes.
    Once the rice is cooked, turn off the heat and let it rest for 10 minutes.
    Fluff up the rice gently with a fork, and drizzle the saffron milk on top.
    Garnish with chopped mint leaves, coriander leaves, and fried onions (if using).
    Serve hot and enjoy the delicious Vegetable Biryani!

        """,  # Replace with actual recipe
        'Dhokla': """Recipe for Dhokla: ...
        In a mixing bowl, add chickpea flour, fine semolina, ginger paste, green chili paste, lemon juice, turmeric powder, baking soda, and salt. Mix well.
    Gradually add water and whisk to make a smooth batter of pouring consistency. Let it rest for 10-15 minutes.
    Grease a steamer plate or a round cake pan with oil.
    Heat water in a steamer or a large pot for steaming the dhokla.
    Just before steaming, add a little more water to the batter if it has thickened during resting.
    Pour the batter into the greased plate or pan and tap gently to spread evenly.
    Place the plate or pan in the steamer or pot and steam on medium-high heat for 12-15 minutes or until the dhokla is cooked and a toothpick inserted in the center comes out clean.
    Once done, turn off the heat and let the dhokla cool for a few minutes.
    In a small pan, heat oil for tempering. Add mustard seeds, cumin seeds, asafoetida, green chilies, and curry leaves. Let them crackle.
    Add water and let it come to a boil.
    Pour the tempering mixture evenly over the steamed dhokla.
    Garnish with chopped coriander leaves.
    Cut the dhokla into squares or diamond shapes.
    Serve the delicious and fluffy dhokla with green chutney, tamarind chutney, or yogurt on the side.
    Enjoy your homemade Dhokla!

        """,  # Replace with actual recipe
        'Dosa': """Recipe for Dosa: ...
    Heat a non-stick dosa tawa or griddle on medium-high heat.
    Once the tawa is hot, sprinkle some water on it and wipe it off with a clean cloth or tissue to cool it slightly.
    Pour a ladleful of dosa batter onto the center of the tawa and spread it out in a circular motion with the back of the ladle to make a thin, even dosa.
    Drizzle some oil or ghee around the edges of the dosa and in the center.
    Cook the dosa on medium-high heat until the edges start to brown and the center is cooked.
    Flip the dosa and cook for a few seconds on the other side.
    Remove the dosa from the tawa and serve hot with coconut chutney, sambar, and/or tomato-onion chutney.
    Enjoy your homemade dosa!

        """,  # Replace with actual recipe
        'Gulab_Jamun': """Recipe for Gulab Jamun: ...
        For Sugar Syrup:

    In a saucepan, combine sugar and water. Place it over medium heat and bring it to a boil.
    Stir the sugar syrup until the sugar dissolves completely.
    Add cardamom pods (if using) and let the syrup simmer for 5-6 minutes until it thickens slightly.
    Add rose water and turn off the heat. Keep the sugar syrup warm.

    For Gulab Jamun:

    In a mixing bowl, combine milk powder, all-purpose flour, ghee or melted butter, milk, baking soda, cardamom powder, and nutmeg powder (if using). Mix well to form a smooth dough.
    If the dough feels dry, add a little more milk, a teaspoon at a time, until it comes together into a soft, pliable dough.
    Divide the dough into small lemon-sized balls and roll them between your palms to make smooth, crack-free balls.
    Heat oil or ghee in a deep frying pan or kadai over medium heat.
    Once the oil or ghee is hot, reduce the heat to low and gently drop the balls into the hot oil or ghee, one by one.
    Fry the balls on low heat, stirring gently, until they turn golden brown and crisp from all sides.
    Remove the fried Gulab Jamun balls using a slotted spoon and transfer them to the warm sugar syrup.
    Let the Gulab Jamuns soak in the sugar syrup for at least 30 minutes to an hour, allowing them to absorb the syrup and become soft and spongy.
    Serve warm Gulab Jamuns with a drizzle of sugar. syrup on top, garnished with chopped nuts (optional).
    Enjoy your homemade Gulab Jamun, a delightful Indian dessert!



        """,  # Replace with actual recipe
        'Idli': """Recipe for Idli: ...
        Instructions:

    Wash the rice and urad dal separately in water multiple times until the water runs clear.
    Soak the rice and urad dal in separate bowls with enough water for 4-5 hours.
    If using fenugreek seeds, soak them along with urad dal.
    After soaking, drain the water and transfer the rice and urad dal to separate grinder jars.
    Grind the urad dal first by adding small amounts of water gradually until you get a smooth and fluffy batter.
    Transfer the urad dal batter to a large mixing bowl.
    Next, grind the rice by adding water as needed until you get a smooth batter with a slightly coarse texture.
    Transfer the rice batter to the same mixing bowl with urad dal batter.
    Add salt to the batter and mix well.
    Cover the bowl with a clean cloth and allow the batter to ferment in a warm place for 8-10 hours or overnight.
    Once fermented, the batter will have risen and become airy.
    Mix the batter gently with a ladle to remove excess air.
    Grease the idli molds with oil or cooking spray.
    Pour the batter into the molds, filling them to 3/4th of their capacity.
    Steam the idlis in a steamer or idli cooker for 10-12 minutes, or until a toothpick inserted into the center of an idli comes out clean.
    Once done, switch off the heat and let the idlis rest for a couple of minutes before removing them from the molds.
    Use a spoon or a butter knife to gently remove the idlis from the molds.
    Serve hot with coconut chutney, sambar, and/or tomato and onion chutney.
    Enjoy your homemade delicious Idlis!



        """,  # Replace with actual recipe
        'Palak_Panner': """Recipe for Palak Panner: ...
        Instructions:

    Heat oil or ghee in a pan or kadai over medium heat.
    Add cumin seeds and let them splutter.
    Add chopped onions and sauté until they turn translucent.
    Add minced garlic, grated ginger, and chopped green chilies (if using). Sauté for a minute.
    Add chopped tomatoes and cook until they turn soft and mushy.
    Add turmeric powder, red chili powder, cumin powder, coriander powder, and salt. Mix well and cook for 2-3 minutes.
    Add chopped spinach leaves and cook until they wilt down.
    Switch off the heat and let the mixture cool for a few minutes.
    Transfer the mixture to a blender or food processor and blend to a smooth puree. You can also use an immersion blender to puree the mixture directly in the pan.
    Return the pureed mixture to the pan and place it over low heat.
    Add paneer cubes and garam masala. Mix gently to coat the paneer with the spinach mixture.
    Cook for another 3-4 minutes, stirring occasionally, until the paneer is heated through.
    If the gravy is too thick, you can add a little water to adjust the consistency.
    Switch off the heat and garnish with a drizzle of ghee or cream (optional).
    Serve hot with roti, naan, or rice.
    Enjoy your delicious Palak Paneer!


        """,  # Replace with actual recipe
        'ButterPaneerMasala': """Recipe for Butter Paneer Masala: ...
        Instructions:

    Heat a pan or kadai over medium heat and add butter.
    Add cumin seeds and let them splutter.
    Add chopped onions and sauté until they turn golden brown.
    Add minced garlic, grated ginger, and slit green chilies (if using). Sauté for a minute.
    Add chopped tomatoes and cook until they turn soft and mushy.
    Add turmeric powder, red chili powder, cumin powder, coriander powder, and salt. Mix well and cook for 2-3 minutes.
    Add cashew nuts and cook for another 2-3 minutes.
    Switch off the heat and let the mixture cool for a few minutes.
    Transfer the mixture to a blender or food processor and blend to a smooth puree. You can also use an immersion blender to puree the mixture directly in the pan.
    Return the pureed mixture to the pan and place it over low heat.
    Add milk or cream and sugar. Mix well and cook for 2-3 minutes.
    Add paneer cubes and garam masala. Mix gently to coat the paneer with the gravy.
    Cook for another 3-4 minutes, stirring occasionally, until the paneer is heated through.
    Switch off the heat and garnish with fresh coriander leaves.
    Serve hot with naan, roti, or rice.
    Enjoy your scrumptious Butter Paneer Masala!




        """,  # Replace with actual recipe
        'Poha': """Recipe for Poha: 
         .\n Rinse the Poha in water for a couple of minutes, then drain and set aside.
         .\n Heat oil in a pan or wok on medium heat.
         .\n Add mustard seeds and cumin seeds, and let them splutter.
         .\n Add chopped onions and sauté until they turn translucent.
         .\n Add diced potatoes and peanuts, and cook until the potatoes are soft.
         .\n Add turmeric powder, red chili powder, sugar, and salt. Mix well.
         .\n Add the rinsed Poha to the pan and mix gently to coat it with the spices.
         .\n Cook for 2-3 minutes, stirring occasionally, until the Poha is heated through.
         .\n Turn off the heat and drizzle lemon juice over the Poha. Mix gently.
         .\n Garnish with chopped coriander leaves.
         .Serve hot and enjoy your delicious Poha!

                """,  # Replace with actual recipe
        'Vada': """Recipe for Vada: ...
        Instructions:

    Wash and soak urad dal in enough water for 4-5 hours or overnight.
    Drain the water and transfer the soaked urad dal to a blender or food processor.
    Add rice flour, cumin seeds, crushed black pepper, grated ginger, chopped green chilies, chopped curry leaves, asafoetida, and salt to the blender.
    Blend everything together to a smooth and thick batter, adding a little water if needed.
    Transfer the batter to a bowl and whisk it well for a few minutes to make it light and fluffy.
    Heat oil in a deep frying pan or kadai over medium heat.
    Drop spoonfuls of the batter into the hot oil and fry on medium-low heat until golden brown and crispy from both sides.
    Remove the vadas from the oil using a slotted spoon and drain on paper towels to remove excess oil.
    Serve hot with coconut chutney, sambar, or tomato chutney.
    Enjoy your crispy and delicious Vadas!


        """,  # Replace with actual recipe
        'Samosa': """Recipe for Samosa: ...

        Instructions:
    For Samosa Pastry:

    In a mixing bowl, combine all-purpose flour, ghee or oil, and a pinch of salt. Mix well.
    Add water gradually, a little at a time, and knead the dough until it becomes smooth and pliable.
    Cover the dough with a damp cloth and let it rest for 15-20 minutes.
    For Samosa Filling:

    Heat oil in a pan over medium heat.
    Add cumin seeds and let them splutter.
    Add chopped onions and green chili. Sauté until onions turn golden brown.
    Add ginger paste, red chili powder, turmeric powder, garam masala powder, and salt. Stir well.
    Add boiled and mashed potatoes and boiled green peas. Mix well.
    Cook the filling for 2-3 minutes, stirring occasionally.
    Add amchur or lemon juice. Mix well and turn off the heat. Let the filling cool.
    For Assembling Samosa:

    Divide the rested dough into small lemon-sized balls and roll them into thin circles.
    Cut the circles into halves to make semi-circles.
    Take a semi-circle and fold it into a cone shape, sealing the edges with water.
    Fill the cone with the cooled potato-pea filling, leaving a little space at the top.
    Apply water on the top edges of the cone and press to seal it, forming a triangle shape.
    Repeat the process with the remaining dough and filling to make all the samosas.
    For Frying Samosa:

    Heat oil in a deep frying pan or kadai over medium heat.
    Gently drop the prepared samosas into the hot oil, a few at a time.
    Fry the samosas on low to medium heat, turning occasionally, until they turn golden brown and crispy from all sides.
    Remove the fried samosas using a slotted spoon and place them on a paper towel-lined plate to absorb excess oil.
    Serve hot and crispy samosas with mint chutney, tamarind chutney, or tomato ketchup for a delicious snack or appetizer!



        """,  # Replace with actual recipe
        'VadaPav': """Recipe for Vada Pav: ...
        Instructions:

    Heat oil in a pan and add mustard seeds and cumin seeds.
    Once the seeds start to splutter, add grated ginger, chopped green chilies, and curry leaves. Saute for a minute.
    Add asafoetida, turmeric powder, and mashed potatoes. Mix well.
    Cook the potato mixture for 2-3 minutes, stirring occasionally. Remove from heat and let it cool.
    In a mixing bowl, whisk together gram flour, turmeric powder, baking soda, salt, and enough water to make a thick batter.
    Heat oil in a deep frying pan or kadai over medium heat.
    Take a portion of the cooled potato mixture and shape it into a small round ball or patty.
    Dip the potato ball or patty into the prepared gram flour batter, coating it evenly from all sides.
    Carefully drop the coated potato ball or patty into the hot oil and fry on medium heat until golden brown and crispy.
    Remove the vada from the oil using a slotted spoon and drain on paper towels to remove excess oil.
    Slice the Pav (burger buns) horizontally and toast them on a hot griddle or pan with a little butter.
    To assemble, spread some green chutney and tamarind chutney on the inner sides of the toasted Pav.
    Place a hot vada in between the Pav and press lightly.
    Serve hot with some thinly sliced onions and chopped cilantro on the side.
    Enjoy your delicious and flavorful Vada Pav!


        """  # Replace with actual recipe
    }

st.markdown("""
    <style>
        .lottie-container > div {
            margin-right: 5px;
            margin-bottom: 5px;
        }
    </style>
""", unsafe_allow_html=True)



with st.container():
    with st.sidebar:
        st_lottie(lottie_coding2, height=100, key="Food2")
    food_types = ["Poha", "Biryani", "Veg Biryani", "Dhokla", "Dosa", "Gulab Jamun", "Samosa", "Idli", "Palak Paneer",
                  "Butter Paneer Masala", "Vada", "Vada Pav"]

    # Create a Streamlit app
    st.title("Available Food Types")



with st.container():
    for food_type in food_types:
        st.write(f"- {food_type}")

    st.write("Welcome to the Food Type Classifier!")
    st.write("Choose any of the above options from the list and insert a file.")

    file = st.file_uploader("Upload an Image file")

    if file is None:
        st.text("Please upload an Image file")
        with st.container():
            st_lottie(lottie_coding4, height=250, key="Food3")

    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        predictions = import_and_predict(image, model)
        predicted_class = class_names[np.argmax(predictions)]
        string = "This image most likely is: " + predicted_class
        st.success(string)

        # Print the recipe for the predicted class
        if predicted_class in recipes:
            recipe = recipes[predicted_class]
            st.info("Recipe: " + recipe)
        else:
            st.warning("Recipe not found for predicted class.")





with st.container():
    st.write("---")
    st.header("Your Valuable feedback appricable!")
    st.write("##")
    contact_form= """
    <form action= "https://formsubmit.co/rohan.saraswat2003@gmail.com" method= "POST">
         <input type="hidden" name="_captcha" value=:false">
         <input type="text" name= "name" placeholder="Your name" required>
         <input type= "text" Feedback="feedback" placeholder="Your email"  required>
         <textarea name="message" placeholder="Your message here" required></textarea>
         <button type= "submit">Send</button>
         
    </form>     
    """

    left_column,right_column= st.columns(2)
    with left_column:
        st.markdown(contact_form,unsafe_allow_html=True)

    with right_column:
        st.empty();





with st.container():
    with st.sidebar:
        members = [
            {"name": "Rohan Saraswat", "email": "rohan.saraswat2003@gmail. com", "linkedin": "https://www.linkedin.com/in/rohan-saraswat-a70a2b225/"},
            {"name": "Saksham Jain", "email": "sakshamgr8online@gmail. com", "linkedin": "https://www.linkedin.com/in/saksham-jain-59b2241a4/"},
            {"name": "Riya Aggarwal ", "email": " riyaaggarwal200326@gmail. com ", "linkedin": "https://www.linkedin.com/in/riya-aggarwal-4a598a200/"}
        ]

        # Define the page title and heading
        st.markdown("<h1 style='font-size:28px'>Team Members</h1>", unsafe_allow_html=True)

        # Iterate over the list of members and display their details
        for member in members:
            st.write(f"Name: {member['name']}")
            st.write(f"Email: {member['email']}")
            st.write(f"LinkedIn: {member['linkedin']}")
            st.write("")

