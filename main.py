from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Text, Optional
from datetime import datetime
from uuid import uuid4 

app = FastAPI()

posts = []

#Modelo
class Post(BaseModel):
    id: Optional[str]
    title: str
    author: str
    content: Text
    created_at: datetime = datetime.now()
    published_at: Optional[datetime]
    published: bool = False

@app.get('/')
def read_root():
    return {"welcome": "Welcome to my API"}

@app.get('/posts')
def get_posts():
    return posts

@app.post('/posts')
def save_post(post: Post):
    post.id = str(uuid4())
    posts.append(post.dict())
    return posts[-1]

@app.get('/posts/{post_id}')
def get_post(post_id: str):
    for post in posts:
        if post["id"] == post_id:
            return post
    raise HTTPException(status_code=404, detail="Post no encontrado")

@app.delete('/posts/{post_id}')
def delete_post(post_id: str):
    for i, post in enumerate(posts):
        if post["id"] == post_id:
            posts.pop(i)
            return {"message" : "Post eliminado exitosamente"}
    raise HTTPException(status_code=404, detail="Post no encontrado")

@app.put('/posts/{post_id}')
def update_post(post_id: str, updated_post: Post):
    for i, post in enumerate(posts):
        if post["id"] == post_id:
            posts["id"]["title"] = update_post.title
            posts["id"]["author"] = update_post.author
            posts["id"]["content"] = update_post.content
            return {"message" : "Post actualizado exitosamente"}
    raise HTTPException(status_code=404, detail="Post no encontrado")