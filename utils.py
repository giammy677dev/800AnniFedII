from neo4j import GraphDatabase
import streamlit as st
from datetime import date
import plotly.express as px
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config

today = date.today()


class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)

    def close(self):
        if self.__driver is not None:
            self.__driver.close()

    def query(self, query, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self.__driver.session(database=db) if db is not None else self.__driver.session()
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response


conn = Neo4jConnection(uri="neo4j+s://1112b239.databases.neo4j.io", user="neo4j", pwd="RGcbWfH9PRM71TkwSnGHb0hILe0btAzBLIKJTD1sQNA")
