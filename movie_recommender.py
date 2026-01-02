
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
import random
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 20px;
    }
    .movie-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s;
    }
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .stat-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        font-weight: bold;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess movie data"""
    try:
        # Load data
        movies_df = pd.read_csv('movies.csv')
        
        # Parse fields
        def parse_field(field):
            if pd.isna(field):
                return []
            if isinstance(field, str):
                # Try to parse as list/dict
                if field.startswith('['):
                    try:
                        parsed = ast.literal_eval(field)
                        if isinstance(parsed, list):
                            if parsed and isinstance(parsed[0], dict):
                                return [item.get('name', str(item)) for item in parsed]
                            return [str(item) for item in parsed]
                        return [str(parsed)]
                    except:
                        pass
                # Comma-separated values
                if ',' in field:
                    return [item.strip() for item in field.split(',')]
                return [field.strip()]
            return [str(field)]
        
        # Parse genres and keywords
        movies_df['genres_list'] = movies_df['Movie_Genre'].apply(parse_field)
        movies_df['keywords_list'] = movies_df['Movie_Keywords'].apply(parse_field)
        
        # Parse cast
        movies_df['cast_list'] = movies_df['Movie_Cast'].apply(
            lambda x: [str(actor).strip() for actor in str(x).split()[:5]] 
            if pd.notna(x) else []
        )
        
        # Parse director
        movies_df['director_list'] = movies_df['Movie_Director'].apply(
            lambda x: [x.strip()] if pd.notna(x) and str(x).strip() else []
        )
        
        # Create combined features
        movies_df['clean_overview'] = movies_df['Movie_Overview'].fillna('').astype(str)
        movies_df['combined_features'] = movies_df.apply(
            lambda row: ' '.join(row['genres_list']) + ' ' + 
                       ' '.join(row['keywords_list']) + ' ' + 
                       ' '.join(row['cast_list']) + ' ' + 
                       ' '.join(row['director_list']) + ' ' + 
                       row['clean_overview'],
            axis=1
        )
        
        # Parse release year
        try:
            movies_df['release_year'] = pd.to_datetime(
                movies_df['Movie_Release_Date'], format='%d-%m-%Y', errors='coerce'
            ).dt.year
        except:
            movies_df['release_year'] = None
        
        return movies_df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# ============================================================================
# RECOMMENDATION FUNCTIONS
# ============================================================================

@st.cache_data
def compute_similarity_matrix(movies_df):
    """Compute similarity matrix"""
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(movies_df['combined_features'].fillna(''))
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_similar_movies(movie_title, movies_df, cosine_sim, n=10):
    """Get similar movies"""
    movie_title_lower = str(movie_title).lower()
    
    # Find matches
    matches = movies_df[movies_df['Movie_Title'].str.lower().str.contains(movie_title_lower, na=False)]
    
    if len(matches) == 0:
        # Try fuzzy matching
        for title in movies_df['Movie_Title'].dropna():
            if movie_title_lower in str(title).lower():
                matches = movies_df[movies_df['Movie_Title'] == title]
                break
    
    if len(matches) == 0:
        return None, "Movie not found"
    
    idx = matches.index[0]
    matched_title = matches.iloc[0]['Movie_Title']
    
    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:n+1]
    
    # Get recommendations
    movie_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]
    
    recommendations = movies_df.iloc[movie_indices][[
        'Movie_Title', 'Movie_Genre', 'Movie_Overview', 
        'Movie_Vote', 'Movie_Popularity', 'release_year'
    ]].copy()
    recommendations['Similarity_Score'] = similarity_scores
    
    return recommendations, matched_title

def get_popular_movies(movies_df, n=20, min_votes=10):
    """Get popular movies using weighted rating"""
    if len(movies_df) == 0:
        return movies_df
    
    # Filter movies with minimum votes
    popular_df = movies_df[movies_df['Movie_Vote_Count'] >= min_votes].copy()
    
    if len(popular_df) == 0:
        popular_df = movies_df.copy()
    
    # Calculate weighted rating (IMDB formula)
    C = popular_df['Movie_Vote'].mean()
    m = popular_df['Movie_Vote_Count'].quantile(0.60)
    
    def weighted_rating(row):
        v = row['Movie_Vote_Count']
        R = row['Movie_Vote']
        if v > 0 and m > 0:
            return (v/(v+m) * R) + (m/(m+v) * C)
        return R
    
    popular_df['weighted_rating'] = popular_df.apply(weighted_rating, axis=1)
    popular_df = popular_df.sort_values('weighted_rating', ascending=False)
    
    return popular_df.head(min(n, len(popular_df)))

def get_movies_by_genre(movies_df, genre, n=20):
    """Get movies by genre"""
    if not genre:
        return movies_df.head(0)
    
    genre_lower = str(genre).lower()
    genre_movies = movies_df[movies_df['genres_list'].apply(
        lambda x: any(genre_lower in str(g).lower() for g in x)
    )].copy()
    
    if len(genre_movies) == 0:
        return genre_movies
    
    genre_movies = genre_movies.sort_values(['Movie_Vote', 'Movie_Popularity'], ascending=False)
    return genre_movies.head(min(n, len(genre_movies)))

def get_random_movies(movies_df, n=10):
    """Get random movies"""
    if len(movies_df) == 0:
        return movies_df
    return movies_df.sample(min(n, len(movies_df)))

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_movie_card(row, show_similarity=False):
    """Create a movie card"""
    title = str(row['Movie_Title']) if pd.notna(row['Movie_Title']) else "Unknown Title"
    
    # Get genres
    if isinstance(row.get('genres_list'), list) and row['genres_list']:
        genres = ', '.join([str(g) for g in row['genres_list'][:3]])
    else:
        genres = str(row.get('Movie_Genre', 'Unknown'))
    
    rating = f"{row['Movie_Vote']:.1f}" if pd.notna(row['Movie_Vote']) else "N/A"
    overview = str(row['Movie_Overview'])[:150] + "..." if pd.notna(row['Movie_Overview']) else "No overview available"
    
    # Similarity score
    similarity_html = ""
    if show_similarity and 'Similarity_Score' in row:
        similarity_html = f"<p><strong>Similarity:</strong> {row['Similarity_Score']:.1%}</p>"
    
    # Year
    year_html = ""
    if 'release_year' in row and pd.notna(row['release_year']):
        year_html = f"<p><strong>Year:</strong> {int(row['release_year'])}</p>"
    
    return f"""
    <div class="movie-card">
        <h4>{title}</h4>
        <p><strong>Genre:</strong> {genres}</p>
        <p><strong>Rating:</strong> ⭐ {rating}/10</p>
        {year_html}
        {similarity_html}
        <p><strong>Overview:</strong> {overview}</p>
    </div>
    """

def create_genre_chart(movies_df):
    """Create genre distribution chart"""
    all_genres = []
    for genres in movies_df['genres_list']:
        if isinstance(genres, list):
            all_genres.extend([str(g) for g in genres])
    
    if not all_genres:
        return go.Figure()
    
    genre_counts = pd.Series(all_genres).value_counts().head(15)
    
    fig = go.Figure(data=[
        go.Bar(
            x=genre_counts.index,
            y=genre_counts.values,
            marker_color=px.colors.sequential.Viridis,
            text=genre_counts.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='Top 15 Movie Genres',
        xaxis_title='Genre',
        yaxis_title='Count',
        xaxis_tickangle=-45,
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400
    )
    
    return fig

def create_rating_chart(movies_df):
    """Create rating distribution chart"""
    fig = go.Figure(data=[
        go.Histogram(
            x=movies_df['Movie_Vote'].dropna(),
            nbinsx=30,
            marker_color='skyblue',
            opacity=0.7
        )
    ])
    
    fig.update_layout(
        title='Distribution of Movie Ratings',
        xaxis_title='Rating',
        yaxis_title='Number of Movies',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400
    )
    
    return fig

def create_popularity_chart(movies_df):
    """Create popularity vs rating chart"""
    fig = go.Figure(data=[
        go.Scatter(
            x=movies_df['Movie_Popularity'],
            y=movies_df['Movie_Vote'],
            mode='markers',
            marker=dict(
                size=movies_df['Movie_Vote_Count'].fillna(1) / 10,
                sizemode='area',
                sizeref=2.*max(movies_df['Movie_Vote_Count'].fillna(1))/40.**2,
                sizemin=4,
                color=movies_df['Movie_Vote'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Rating")
            ),
            text=movies_df['Movie_Title'],
            hoverinfo='text+x+y'
        )
    ])
    
    fig.update_layout(
        title='Popularity vs Rating',
        xaxis_title='Popularity',
        yaxis_title='Rating',
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400
    )
    
    return fig

def create_wordcloud_chart(movies_df):
    """Create word cloud from overviews"""
    text = ' '.join(movies_df['clean_overview'].dropna().astype(str))
    
    if not text.strip():
        return go.Figure()
    
    try:
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=100,
            contour_width=3,
            contour_color='steelblue'
        ).generate(text)
        
        wordcloud_array = wordcloud.to_array()
        
        fig = go.Figure(data=[
            go.Image(z=wordcloud_array)
        ])
        
        fig.update_layout(
            title='Word Cloud of Movie Overviews',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=0, r=0, t=40, b=0),
            height=400
        )
        
        return fig
    except:
        return go.Figure()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie Recommendation System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem;">Discover similar movies, explore popular titles, and find films by genre!</p>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('Loading and processing movie data...'):
        movies_df = load_and_preprocess_data()
    
    if movies_df.empty:
        st.error("Failed to load movie data. Please check the data file.")
        return
    
    # Compute similarity matrix
    with st.spinner('Computing movie similarities...'):
        cosine_sim = compute_similarity_matrix(movies_df)
    
    # Get unique genres
    all_genres = []
    for genres in movies_df['genres_list']:
        if isinstance(genres, list):
            all_genres.extend([str(g) for g in genres])
    unique_genres = sorted(set([g for g in all_genres if g]))
    
    # Sidebar with statistics
    with st.sidebar:
        st.markdown("### 📊 Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Movies", f"{len(movies_df):,}")
            st.metric("Avg Rating", f"{movies_df['Movie_Vote'].mean():.1f}/10")
        
        with col2:
            st.metric("Genres", len(unique_genres))
            if 'release_year' in movies_df.columns and movies_df['release_year'].notna().any():
                min_year = int(movies_df['release_year'].min())
                max_year = int(movies_df['release_year'].max())
                st.metric("Year Range", f"{min_year}-{max_year}")
        
        st.divider()
        
        # Quick search
        st.markdown("### 💡 Quick Search")
        sample_movies = ["Star Wars", "The Dark Knight", "Forrest Gump", "Toy Story", "The Matrix"]
        for movie in sample_movies:
            if st.button(f"🔍 {movie}", use_container_width=True):
                st.session_state.search_input = movie
                st.session_state.active_tab = "Search"
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Search & Recommend", 
        "🔥 Popular Movies", 
        "🎭 Browse by Genre", 
        "📊 Statistics", 
        "🎲 Random Movies"
    ])
    
    # Tab 1: Search and Recommendations
    with tab1:
        st.header("🔍 Search & Recommend Movies")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            # Initialize session state
            if 'search_input' not in st.session_state:
                st.session_state.search_input = ""
            
            movie_search = st.text_input(
                "Enter a movie title",
                value=st.session_state.get('search_input', ''),
                placeholder="e.g., 'Star Wars', 'Inception'",
                key="search_input"
            )
            
            n_recommendations = st.slider(
                "Number of recommendations",
                min_value=5,
                max_value=20,
                value=10,
                help="How many similar movies to show"
            )
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                search_btn = st.button("🔎 Find Similar Movies", type="primary", use_container_width=True)
            with col_btn2:
                if st.button("🔄 Clear", use_container_width=True):
                    st.session_state.search_input = ""
                    st.rerun()
            
            # Movie suggestions
            if movie_search:
                movie_titles = movies_df['Movie_Title'].dropna().unique().tolist()
                matches = [title for title in movie_titles if movie_search.lower() in title.lower()]
                if matches:
                    st.markdown("### 🎯 Suggestions")
                    for match in matches[:5]:
                        if st.button(f"🎬 {match}", key=f"suggest_{match}", use_container_width=True):
                            st.session_state.search_input = match
                            st.rerun()
        
        with col2:
            st.markdown("### 🎬 Recommendations")
            
            if search_btn and movie_search:
                with st.spinner(f'Finding movies similar to "{movie_search}"...'):
                    recommendations, matched_title = get_similar_movies(
                        movie_search, movies_df, cosine_sim, n_recommendations
                    )
                
                if recommendations is not None and not recommendations.empty:
                    st.success(f"✅ Found {len(recommendations)} movies similar to **{matched_title}**")
                    
                    # Display recommendations
                    for _, row in recommendations.iterrows():
                        st.markdown(create_movie_card(row, show_similarity=True), unsafe_allow_html=True)
                    
                    # Similarity chart
                    st.subheader("📈 Similarity Scores")
                    fig = go.Figure(data=[
                        go.Bar(
                            x=recommendations['Movie_Title'],
                            y=recommendations['Similarity_Score'],
                            marker_color='indianred',
                            text=[f"{s:.1%}" for s in recommendations['Similarity_Score']],
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        xaxis_title='Movies',
                        yaxis_title='Similarity Score',
                        xaxis_tickangle=-45,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif recommendations is None:
                    st.error(f"❌ Movie '{movie_search}' not found. Please try another title.")
            else:
                st.info("👈 Enter a movie title and click 'Find Similar Movies' to get recommendations")
    
    # Tab 2: Popular Movies
    with tab2:
        st.header("🔥 Popular Movies")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_popular = st.slider(
                "Number of movies to show",
                min_value=10,
                max_value=30,
                value=15,
                key="popular_slider"
            )
            
            min_votes = st.slider(
                "Minimum votes",
                min_value=0,
                max_value=100,
                value=10,
                help="Movies with fewer votes will be excluded"
            )
            
            popular_btn = st.button("🎬 Show Popular Movies", type="primary", use_container_width=True)
        
        with col2:
            if popular_btn:
                with st.spinner('Finding popular movies...'):
                    popular_movies = get_popular_movies(movies_df, n_popular, min_votes)
                
                if not popular_movies.empty:
                    st.success(f"✅ Showing top {len(popular_movies)} popular movies")
                    
                    for _, row in popular_movies.iterrows():
                        st.markdown(create_movie_card(row), unsafe_allow_html=True)
                    
                    # Popularity chart
                    st.subheader("📊 Popularity Chart")
                    fig = go.Figure(data=[
                        go.Bar(
                            x=popular_movies['Movie_Title'],
                            y=popular_movies['Movie_Vote'],
                            marker_color='lightgreen',
                            text=popular_movies['Movie_Vote'].round(1),
                            textposition='auto'
                        )
                    ])
                    
                    fig.update_layout(
                        title=f'Top {n_popular} Popular Movies by Rating',
                        xaxis_title='Movie Title',
                        yaxis_title='Rating',
                        xaxis_tickangle=-45,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No popular movies found with the current criteria.")
            else:
                st.info("👈 Click 'Show Popular Movies' to see top-rated films")
    
    # Tab 3: Movies by Genre
    with tab3:
        st.header("🎭 Browse Movies by Genre")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_genre = st.selectbox(
                "Select a genre",
                options=unique_genres[:50] if len(unique_genres) > 50 else unique_genres,
                index=0,
                help=f"Total genres available: {len(unique_genres)}"
            )
            
            n_genre_movies = st.slider(
                "Number of movies to show",
                min_value=10,
                max_value=30,
                value=15,
                key="genre_slider"
            )
            
            genre_btn = st.button("🎭 Show Genre Movies", type="primary", use_container_width=True)
        
        with col2:
            if genre_btn and selected_genre:
                with st.spinner(f'Finding {selected_genre} movies...'):
                    genre_movies = get_movies_by_genre(movies_df, selected_genre, n_genre_movies)
                
                if not genre_movies.empty:
                    st.success(f"✅ Found {len(genre_movies)} {selected_genre} movies")
                    
                    for _, row in genre_movies.iterrows():
                        st.markdown(create_movie_card(row), unsafe_allow_html=True)
                    
                    # Genre statistics
                    st.subheader("📊 Genre Statistics")
                    
                    col_stats1, col_stats2 = st.columns(2)
                    
                    with col_stats1:
                        avg_rating = genre_movies['Movie_Vote'].mean()
                        st.metric(f"Avg Rating", f"{avg_rating:.1f}/10")
                    
                    with col_stats2:
                        total_movies = len(genre_movies)
                        st.metric(f"Total Movies", total_movies)
                    
                    # Box plot
                    fig = go.Figure(data=[
                        go.Box(
                            y=genre_movies['Movie_Vote'].dropna(),
                            name=selected_genre,
                            marker_color='lightblue'
                        )
                    ])
                    
                    fig.update_layout(
                        title=f'Rating Distribution for {selected_genre} Movies',
                        yaxis_title='Rating',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"⚠️ No movies found for genre: {selected_genre}")
            else:
                st.info("👈 Select a genre and click 'Show Genre Movies'")
    
    # Tab 4: Statistics
    with tab4:
        st.header("📊 Dataset Statistics")
        
        # Quick stats
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Movies", len(movies_df))
        with col2:
            st.metric("Average Rating", f"{movies_df['Movie_Vote'].mean():.2f}/10")
        with col3:
            st.metric("Highest Rating", f"{movies_df['Movie_Vote'].max():.2f}/10")
        with col4:
            st.metric("Lowest Rating", f"{movies_df['Movie_Vote'].min():.2f}/10")
        
        # Charts
        st.subheader("📈 Visualizations")
        
        col_chart1, col_chart2 = st.columns(2)
        with col_chart1:
            st.plotly_chart(create_genre_chart(movies_df), use_container_width=True)
        with col_chart2:
            st.plotly_chart(create_rating_chart(movies_df), use_container_width=True)
        
        col_chart3, col_chart4 = st.columns(2)
        with col_chart3:
            st.plotly_chart(create_popularity_chart(movies_df), use_container_width=True)
        with col_chart4:
            st.plotly_chart(create_wordcloud_chart(movies_df), use_container_width=True)
        
        # Top movies and genres
        st.subheader("📋 Detailed Statistics")
        
        col_stats1, col_stats2 = st.columns(2)
        with col_stats1:
            st.markdown("### 🎭 Top 10 Genres")
            all_genres_list = []
            for genres in movies_df['genres_list']:
                if isinstance(genres, list):
                    all_genres_list.extend([str(g) for g in genres])
            genre_counts = Counter(all_genres_list)
            top_genres = pd.DataFrame(genre_counts.most_common(10), columns=['Genre', 'Count'])
            st.dataframe(top_genres, use_container_width=True, hide_index=True)
        
        with col_stats2:
            st.markdown("### ⭐ Top 10 Rated Movies")
            top_rated = movies_df.nlargest(10, 'Movie_Vote')[['Movie_Title', 'Movie_Vote', 'Movie_Genre']]
            st.dataframe(top_rated, use_container_width=True, hide_index=True)
    
    # Tab 5: Random Movies
    with tab5:
        st.header("🎲 Random Movie Discovery")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            n_random = st.slider(
                "Number of random movies",
                min_value=5,
                max_value=20,
                value=10,
                key="random_slider"
            )
            
            random_btn = st.button("🎲 Get Random Movies", type="primary", use_container_width=True)
            
            st.markdown("""
            ### 💡 Perfect for when you can't decide!
            
            **Features:**
            - Random selection from all movies
            - Great for discovering hidden gems
            - Perfect for movie nights
            """)
        
        with col2:
            if random_btn:
                with st.spinner('Finding random movies...'):
                    random_movies = get_random_movies(movies_df, n_random)
                
                if not random_movies.empty:
                    st.success(f"✅ Found {len(random_movies)} random movies for you!")
                    
                    for _, row in random_movies.iterrows():
                        st.markdown(create_movie_card(row), unsafe_allow_html=True)
                    
                    # Random movie stats
                    st.subheader("🎯 Your Random Selection")
                    col_rand1, col_rand2 = st.columns(2)
                    with col_rand1:
                        avg_rating = random_movies['Movie_Vote'].mean()
                        st.metric("Average Rating", f"{avg_rating:.1f}/10")
                    with col_rand2:
                        genres_in_random = set()
                        for genres in random_movies['genres_list']:
                            if isinstance(genres, list):
                                genres_in_random.update([str(g) for g in genres])
                        st.metric("Unique Genres", len(genres_in_random))
                else:
                    st.warning("⚠️ Could not find random movies.")
            else:
                st.info("👈 Click 'Get Random Movies' for surprise recommendations!")
    
    # Footer
    st.divider()
    st.markdown(f"""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🎬 Movie Recommendation System • Built with ❤️ using Streamlit</p>
        <p>📊 Using TMDB 5000 Movie Dataset • {len(movies_df):,} movies available</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
