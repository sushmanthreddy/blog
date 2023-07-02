+++
title =  "How Did I Create This Blog ?"
tags = ["hugo", "themes", "github-pages", "blog"]
date = "2020-11-20"
+++

## Intro :

I should have created this blog earlier itself but after two years of gradutaion ,here iam creating it 
this blog page is hell out of ride ,i dont knew anything about web dev ,thanks to hugo framework,with help 
of some youtube videos and mainak's blog ,i made my own blog .
## The Difficult Part :

I'll divide it into steps :

1. Install Hugo ; I'll skip this step, just google "install hugo {your OS i.e. Ubuntu, Windows}" and follow the steps.

```
  $ brew install hugo 

```
installs hugo in your mac

2. Then you'll need to create a new site and download a theme. I simply cloned a theme into the themes directory.Copy paste the following commands in your terminal

```
# Create new site
$ hugo new site mainaks_site
$ cd mainaks_site
$ cd themes
# Download the theme of your choice. The given link is for the theme I used.
$ git clone https://github.com/kimcc/hugo-theme-noteworthy.git

```
3. Go into ```./sushmanthblog/themes/noteworthy/exampleSite/``` and copy paste the files in that directory (like config.toml, content etc.) into ```./sushmanthblog/``` . Replace existing files if asked.

4. One important thing to remember is to edit the ```config.toml``` file in ```./sushmanthblog/```. 

    * Open the ```config.toml``` file and set the themedir as the directory to the cloned theme which was ```./themes/```in my case.
    * Set theme as the name of the cloned folder, which was ```noteworthy``` in my case because I had renamed the cloned folder from ```hugo-theme-noteworthy``` to just ```noteworthy```. 
    * Set the baseURL as ```https://username.github.io/``` where username is your Github username, and set blog title.
    * Set the other params to you wish.
    * make two github repo and blog and repo named with ur user name and endeing with username.github.io
    
    There's one more change that needs to be made in the ```config.toml``` file, which I'm saving for later. Let's run the website locally first.

## (Local) Moment of truth :

Open ```./sushmanthblog/``` directory in your terminal and run ```hugo serve```. If all goes well.

Now open ```http://localhost:1313/``` in your browser, you will find your blog up and running :)

## Edit Content and Add Images :
Ok, this step should be easy. All you need to do now is edit the markdown files in ```./sushmanthblog/content/posts/``` directory.

 If you want to add images in your post, create a folder named ```images``` in the ```./posts/``` directory and store your images there, then use the following syntax in your markdown file to insert the images.
```
![](../images/doggo_2.png)
```
## Build Site and Upload Files to Github :
Navigate into ```./sushmanthblog/``` and run ```$ hugo``` in your terminal, wait for hugo to finish building your site. Now you should see an output that looks like the following image.

Now we need to upload all static files from our public file to ```username.github.io``` and main frame work lies in the blog all static files are hosted in ```github.io```.





Well, thats how I got this blog up and running. I'll upload many more posts in the upcoming weeks, stay tuned for that :)
