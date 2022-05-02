import React from "react"
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faTwitter, faGithub, faLinkedin } from '@fortawesome/free-brands-svg-icons'


const Footer = () => (
    <footer className="bg-gray-100 py-16 text-gray-800">
        <div className="container">
            <div className="text-center">
                <div className="flex gap-4 mb-6 justify-center">
                    <a href="https://twitter.com/beraamit_" target="_blank" rel="noreferrer" className="text-gray-800 text-xl hover:text-indigo-600">
                        <FontAwesomeIcon icon={faTwitter} />
                    </a>
                    <a href="https://github.com/amitalokbera" target="_blank" rel="noreferrer" className="text-gray-800 text-xl hover:text-indigo-600">
                        <FontAwesomeIcon icon={faGithub} />
                    </a>
                    <a href="https://linkedin.com/in/amit-bera-1a7b541b1" target="_blank" rel="noreferrer" className="text-gray-800 text-xl hover:text-indigo-600">
                        <FontAwesomeIcon icon={faLinkedin} />
                    </a>
                </div>
                <p className="mb-0">&copy; Amit Bera { new Date().getFullYear() }</p>
            </div>
        </div>
    </footer>
)

export default Footer