import streamlit as st
import requests
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Загрузка данных
with open('cv.pkl', 'rb') as f:
    cv = pickle.load(f)
    
with open('indices.pkl', 'rb') as f:
    indices = pickle.load(f)

with open('movies_list.pkl', 'rb') as f:
    filtered_df = pickle.load(f)

def get_movie_details(movie_id):
    response = requests.get(
        f'https://api.themoviedb.org/3/movie/{movie_id}?api_key=ccaa561c8bbe4cff97a35aae1cbc48ca&language=en-US'
    )
    data = response.json()
    poster_url = "https://image.tmdb.org/t/p/w500/" + data['poster_path']
    overview = data['overview']
    release_date = data['release_date']
    runtime = data['runtime']
    return poster_url, overview, release_date, runtime

# Функция для получения рекомендаций
def get_recommendations(movie_title):
    idx = filtered_df[filtered_df['title'].str.contains(movie_title, case=False)].index
    if len(idx) == 0:
        st.error("Фильм не найден. Попробуйте другое название.")
        return []
    
    idx = idx[0]  # Получаем индекс первого совпадения
    similarity_scores = cosine_similarity(cv[idx], cv).flatten()
    similar_movie_indices = similarity_scores.argsort()[-7:-1][::-1]

    return filtered_df['title'].iloc[similar_movie_indices].tolist()

# Интерфейс Streamlit
st.title("🎬 Система рекомендации фильмов")

movie_title = st.selectbox("Введите название фильма:", 
                            filtered_df['title'].values)

if st.button("Получить рекомендации"):
    recommendations = get_recommendations(movie_title)
    
    if recommendations:
        st.subheader("Рекомендованные фильмы:")
        for title in recommendations:

            movie_id = filtered_df[filtered_df['title'] == title]['id'].values[0]
            poster_url, overview, release_date, runtime = get_movie_details(movie_id)
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(poster_url, width=150)
            with col2:
                st.write(f"**Название:** {title}")
                st.write(f"**Описание:** {overview}")
                st.write(f"**Дата выхода:** {release_date}")
                st.write(f"**Время:** {runtime} минут")
            st.write("---")
