import React from "react"
import Layout from "../src/components/layout"
import Seo from "../src/components/seo"
import BlogCard from "../src/components/BlogCard/BlogCard"
import Heading from "../src/components/Heading"
import { getAllPosts } from "../lib/blog"

const Index = ({ posts }) => {
  const Posts = () => {
    return posts.map((post, index) => (
      <div key={index} style={{ marginBottom: 40 }}>
        <BlogCard
          title={post.frontmatter.title}
          slug={post.frontmatter.slug}
          date={post.frontmatter.date}
        />
      </div>
    ))
  }

  return (
    <Layout>
      <Seo title="Home | Machine Learning Engineer in Mumbai, India" />
      <div className="container">
        <div className="py-10 md:py-32 md:w-2/3">
          <h2 className="text-3xl md:text-5xl font-bold mb-2">
            Hey, I'm Amit{" "}
            <span role="img" aria-label="wave">
              👋
            </span>
          </h2>
          <h1 className="text-xl md:text-2xl font-bold mb-2">
            Full Stack Engineer based in Mumbai, India
          </h1>
        </div>
        <div className="mt-16">
          <Heading level={2}>Blog Posts</Heading>
          <Posts />
        </div>
      </div>
    </Layout>
  )
}

export default Index

export async function getStaticProps() {
  const posts = getAllPosts().sort(
    (a, b) => new Date(b.frontmatter.date) - new Date(a.frontmatter.date)
  )
  return {
    props: {
      posts,
    },
  }
}
